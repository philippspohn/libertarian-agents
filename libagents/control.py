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
  output_tokens       INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
  cost_usd            REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS usage_env_profile ON usage (env, profile);
"""

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    stop_reason: Optional[str] = None


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


# --------------------------------------------------------------------------
# runners


def _runner(row: sqlite3.Row) -> Runner:
    return Runner(
        env=row["env"],
        profile=row["profile"],
        state=row["state"],
        config=RunnerConfig.model_validate_json(row["config"]),
        wake_at=row["wake_at"],
        stop_reason=row["stop_reason"],
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


def get_runner(env: str, profile: str) -> Runner:
    with connect() as c:
        row = c.execute("SELECT * FROM runners WHERE env=? AND profile=?", (env, profile)).fetchone()
    if row is None:
        raise KeyError(f"no such runner: {env}/{profile}")
    return _runner(row)


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
    stop_reason: Optional[str] = None,
) -> None:
    with connect() as c:
        c.execute(
            "UPDATE runners SET state=?, wake_at=?, stop_reason=?, updated_at=? WHERE env=? AND profile=?",
            (state, wake_at, stop_reason, _now(), env, profile),
        )


def delete_runner(env: str, profile: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM runners WHERE env=? AND profile=?", (env, profile))


# --------------------------------------------------------------------------
# usage ledger


def record_usage(env: str, profile: str, model: str, usage: UsageRow) -> None:
    with connect() as c:
        c.execute(
            """INSERT INTO usage (env, profile, ts, model, input_tokens,
                                  cached_input_tokens, output_tokens,
                                  reasoning_tokens, cost_usd)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                env,
                profile,
                _now(),
                model,
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.output_tokens,
                usage.reasoning_tokens,
                usage.cost_usd,
            ),
        )


def usage_for(env: str, profile: Optional[str] = None) -> UsageRow:
    q = """SELECT COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(cached_input_tokens),0) ci,
                  COALESCE(SUM(output_tokens),0) o, COALESCE(SUM(reasoning_tokens),0) r,
                  COALESCE(SUM(cost_usd),0) c
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
        output_tokens=row["o"],
        reasoning_tokens=row["r"],
        cost_usd=row["c"],
    )


def usage_breakdown(env: str) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            """SELECT profile, model, SUM(input_tokens) i, SUM(cached_input_tokens) ci,
                      SUM(output_tokens) o, SUM(reasoning_tokens) r, SUM(cost_usd) c,
                      COUNT(*) calls
               FROM usage WHERE env=? GROUP BY profile, model ORDER BY profile""",
            (env,),
        ).fetchall()
    return [dict(r) for r in rows]


def wait_until(deadline: float, poll: float = 0.25) -> None:
    while time.time() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.time())))
