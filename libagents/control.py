"""Host-side control plane.

This SQLite DB lives OUTSIDE the sandbox and holds everything an agent must
not be able to change: runner state, model + budget config, and the usage
ledger. Environment content (files, memory, the message board) is not
duplicated here -- the filesystem is the source of truth for that.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from . import paths
from .models import Budgets, EnvConfig, RunnerConfig, RunnerState, UsageRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS environments (
  name       TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  config     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runners (
  env         TEXT NOT NULL,
  profile     TEXT NOT NULL,
  state       TEXT NOT NULL DEFAULT 'inactive',
  config      TEXT NOT NULL,
  wake_at     REAL,
  wake_after_id INTEGER NOT NULL DEFAULT 0,
  stop_reason TEXT,
  updated_at  TEXT NOT NULL,
  PRIMARY KEY (env, profile)
);
CREATE TABLE IF NOT EXISTS usage (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  env                 TEXT NOT NULL,
  profile             TEXT NOT NULL,
  ts                  TEXT NOT NULL,
  model               TEXT NOT NULL,
  input_tokens        INTEGER NOT NULL DEFAULT 0,
  cached_input_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens       INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
  cost_usd            REAL NOT NULL DEFAULT 0,
  cost_known          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS usage_env_profile ON usage (env, profile);
CREATE TABLE IF NOT EXISTS schema_migrations (
  name       TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);
"""

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migrate_config_scope(conn: sqlite3.Connection) -> None:
    """Move legacy environment goal/summary settings onto each runner.

    Removing the legacy keys makes this migration one-shot, so a deliberately
    cleared per-runner goal is never repopulated later.
    """
    for row in conn.execute("SELECT name, config FROM environments").fetchall():
        env_cfg = json.loads(row["config"])
        legacy_goal = env_cfg.pop("goal", None)
        legacy_summary_provider = env_cfg.pop("summary_provider", None)
        legacy_summary_model = env_cfg.pop("summary_model", None)
        if not any(v is not None for v in (legacy_goal, legacy_summary_provider, legacy_summary_model)):
            continue
        runner_rows = conn.execute(
            "SELECT profile, config FROM runners WHERE env=?", (row["name"],)
        ).fetchall()
        for runner_row in runner_rows:
            runner_cfg = json.loads(runner_row["config"])
            if legacy_goal is not None and "goal" not in runner_cfg:
                runner_cfg["goal"] = legacy_goal
            if legacy_summary_provider is not None and "summary_provider" not in runner_cfg:
                runner_cfg["summary_provider"] = legacy_summary_provider
            if legacy_summary_model is not None and "summary_model" not in runner_cfg:
                runner_cfg["summary_model"] = legacy_summary_model
            conn.execute(
                "UPDATE runners SET config=? WHERE env=? AND profile=?",
                (json.dumps(runner_cfg), row["name"], runner_row["profile"]),
            )
        conn.execute(
            "UPDATE environments SET config=? WHERE name=?",
            (json.dumps(env_cfg), row["name"]),
        )


def _migrate_runner_prompt_and_tools(conn: sqlite3.Connection) -> None:
    """Remove the retired append-only prompt layer and give existing runners
    the new channel counterpart. The migration table makes this one-shot, so
    an operator can subsequently disable leave_channel deliberately."""
    name = "runner-prompt-and-leave-channel-v1"
    if conn.execute(
        "SELECT 1 FROM schema_migrations WHERE name=?", (name,)
    ).fetchone():
        return
    for row in conn.execute("SELECT env, profile, config FROM runners").fetchall():
        config = json.loads(row["config"])
        changed = config.pop("operator_instructions", None) is not None
        tools = config.get("tools")
        if isinstance(tools, list) and "join_channel" in tools and "leave_channel" not in tools:
            tools.insert(tools.index("join_channel") + 1, "leave_channel")
            changed = True
        if changed:
            conn.execute(
                "UPDATE runners SET config=? WHERE env=? AND profile=?",
                (json.dumps(config), row["env"], row["profile"]),
            )
    conn.execute(
        "INSERT INTO schema_migrations (name, applied_at) VALUES (?,?)",
        (name, _now()),
    )


def _backfill_gpt56_pricing(conn: sqlite3.Connection) -> None:
    """Price existing unknown GPT-5.6 rows using the new built-in table."""
    name = "gpt56-pricing-backfill-v1"
    if conn.execute(
        "SELECT 1 FROM schema_migrations WHERE name=?", (name,)
    ).fetchone():
        return
    from .pricing import cost

    rows = conn.execute(
        """SELECT id, model, input_tokens, cached_input_tokens,
                  cache_write_tokens, output_tokens
           FROM usage WHERE cost_known=0"""
    ).fetchall()
    for row in rows:
        estimate = cost(
            row["model"], row["input_tokens"], row["cached_input_tokens"],
            row["output_tokens"], row["cache_write_tokens"],
        )
        if estimate is not None:
            conn.execute(
                "UPDATE usage SET cost_usd=?, cost_known=1 WHERE id=?",
                (estimate, row["id"]),
            )
    conn.execute(
        "INSERT INTO schema_migrations (name, applied_at) VALUES (?,?)",
        (name, _now()),
    )


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = paths.control_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        with _lock:
            conn.executescript(SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(runners)")}
            if "wake_after_id" not in columns:
                conn.execute(
                    "ALTER TABLE runners ADD COLUMN wake_after_id INTEGER NOT NULL DEFAULT 0"
                )
            usage_columns = {r[1] for r in conn.execute("PRAGMA table_info(usage)")}
            if "cache_write_tokens" not in usage_columns:
                conn.execute(
                    "ALTER TABLE usage ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0"
                )
            if "cost_known" not in usage_columns:
                # Existing zero costs are ambiguous, so migrate them
                # conservatively as unknown rather than claiming they were free.
                conn.execute(
                    "ALTER TABLE usage ADD COLUMN cost_known INTEGER NOT NULL DEFAULT 0"
                )
            _migrate_config_scope(conn)
            _migrate_runner_prompt_and_tools(conn)
            _backfill_gpt56_pricing(conn)
            yield conn
            conn.commit()
    finally:
        conn.close()


@dataclass
class Runner:
    env: str
    profile: str
    state: RunnerState
    config: RunnerConfig
    wake_at: Optional[float] = None
    wake_after_id: int = 0
    stop_reason: Optional[str] = None
    updated_at: str = ""


# --------------------------------------------------------------------------
# environments


def create_env(name: str, config: EnvConfig) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO environments (name, created_at, config) VALUES (?,?,?)",
            (name, _now(), config.model_dump_json()),
        )


def env_exists(name: str) -> bool:
    with connect() as c:
        return c.execute("SELECT 1 FROM environments WHERE name=?", (name,)).fetchone() is not None


def get_env(name: str) -> EnvConfig:
    with connect() as c:
        row = c.execute("SELECT config FROM environments WHERE name=?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"no such environment: {name}")
    return EnvConfig.model_validate_json(row["config"])


def set_env(name: str, config: EnvConfig) -> None:
    with connect() as c:
        c.execute("UPDATE environments SET config=? WHERE name=?", (config.model_dump_json(), name))


def list_envs() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT name, created_at, config FROM environments ORDER BY name").fetchall()
    return [
        {"name": r["name"], "created_at": r["created_at"], "config": json.loads(r["config"])}
        for r in rows
    ]


def delete_env(name: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM environments WHERE name=?", (name,))
        c.execute("DELETE FROM runners WHERE env=?", (name,))
        c.execute("DELETE FROM usage WHERE env=?", (name,))


# --------------------------------------------------------------------------
# runners


def _runner(row: sqlite3.Row) -> Runner:
    return Runner(
        env=row["env"],
        profile=row["profile"],
        state=row["state"],
        config=RunnerConfig.model_validate_json(row["config"]),
        wake_at=row["wake_at"],
        wake_after_id=row["wake_after_id"],
        stop_reason=row["stop_reason"],
        updated_at=row["updated_at"],
    )


def upsert_runner(env: str, profile: str, config: RunnerConfig) -> None:
    with connect() as c:
        c.execute(
            """INSERT INTO runners (env, profile, state, config, updated_at)
               VALUES (?,?,'inactive',?,?)
               ON CONFLICT (env, profile) DO UPDATE SET config=excluded.config,
                                                        updated_at=excluded.updated_at""",
            (env, profile, config.model_dump_json(), _now()),
        )


def add_input_budget(env: str, profile: str, tokens: int) -> RunnerConfig:
    """Atomically add to one runner's input-token cap.

    This intentionally updates only the nested budget field so an active
    runner adjustment cannot overwrite unrelated configuration changes.
    """
    if tokens <= 0:
        raise ValueError("budget increment must be positive")
    with connect() as c:
        row = c.execute(
            "SELECT config FROM runners WHERE env=? AND profile=?", (env, profile)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such runner: {env}/{profile}")
        config = RunnerConfig.model_validate_json(row["config"])
        config.budgets.input_tokens += tokens
        c.execute(
            "UPDATE runners SET config=?, updated_at=? WHERE env=? AND profile=?",
            (config.model_dump_json(), _now(), env, profile),
        )
    return config


def get_runner(env: str, profile: str) -> Runner:
    with connect() as c:
        row = c.execute("SELECT * FROM runners WHERE env=? AND profile=?", (env, profile)).fetchone()
    if row is None:
        raise KeyError(f"no such runner: {env}/{profile}")
    return _runner(row)


def runner_exists(env: str, profile: str) -> bool:
    with connect() as c:
        return c.execute(
            "SELECT 1 FROM runners WHERE env=? AND profile=?", (env, profile)
        ).fetchone() is not None


def list_runners(env: Optional[str] = None) -> list[Runner]:
    q = "SELECT * FROM runners"
    args: tuple = ()
    if env:
        q += " WHERE env=?"
        args = (env,)
    with connect() as c:
        return [_runner(r) for r in c.execute(q + " ORDER BY profile", args).fetchall()]


def set_state(
    env: str,
    profile: str,
    state: RunnerState,
    *,
    wake_at: Optional[float] = None,
    wake_after_id: int = 0,
    stop_reason: Optional[str] = None,
) -> None:
    with connect() as c:
        c.execute(
            "UPDATE runners SET state=?, wake_at=?, wake_after_id=?, stop_reason=?, updated_at=? "
            "WHERE env=? AND profile=?",
            (state, wake_at, wake_after_id, stop_reason, _now(), env, profile),
        )


def delete_runner(env: str, profile: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM runners WHERE env=? AND profile=?", (env, profile))
        c.execute("DELETE FROM usage WHERE env=? AND profile=?", (env, profile))


# --------------------------------------------------------------------------
# usage ledger


def record_usage(env: str, profile: str, model: str, usage: UsageRow) -> None:
    with connect() as c:
        c.execute(
            """INSERT INTO usage (env, profile, ts, model, input_tokens,
                                  cached_input_tokens, cache_write_tokens, output_tokens,
                                  reasoning_tokens, cost_usd, cost_known)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                env,
                profile,
                _now(),
                model,
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.cache_write_tokens,
                usage.output_tokens,
                usage.reasoning_tokens,
                usage.cost_usd,
                int(usage.cost_known),
            ),
        )


def usage_for(env: str, profile: Optional[str] = None) -> UsageRow:
    q = """SELECT COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(cached_input_tokens),0) ci,
                  COALESCE(SUM(cache_write_tokens),0) cw,
                  COALESCE(SUM(output_tokens),0) o, COALESCE(SUM(reasoning_tokens),0) r,
                  COALESCE(SUM(cost_usd),0) c, COUNT(*) n,
                  CASE WHEN COUNT(*)=0 THEN 1 ELSE MIN(cost_known) END k
           FROM usage WHERE env=?"""
    args: tuple = (env,)
    if profile:
        q += " AND profile=?"
        args = (env, profile)
    with connect() as c:
        row = c.execute(q, args).fetchone()
    return UsageRow(
        input_tokens=row["i"],
        cached_input_tokens=row["ci"],
        cache_write_tokens=row["cw"],
        output_tokens=row["o"],
        reasoning_tokens=row["r"],
        cost_usd=row["c"],
        cost_known=bool(row["k"]),
    )


def usage_breakdown(env: str) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            """SELECT profile, model, SUM(input_tokens) i, SUM(cached_input_tokens) ci,
                      SUM(cache_write_tokens) cw,
                      SUM(output_tokens) o, SUM(reasoning_tokens) r, SUM(cost_usd) c,
                      MIN(cost_known) k,
                      COUNT(*) calls
               FROM usage WHERE env=? GROUP BY profile, model ORDER BY profile""",
            (env,),
        ).fetchall()
    return [dict(r) for r in rows]


def wait_until(deadline: float, poll: float = 0.25) -> None:
    while time.time() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.time())))
