"""Tests for the parts where a silent bug would be expensive: path escapes,
unread bookkeeping, the memory.md limit, and the tool-call/result pairing that
providers reject if we get it wrong."""

from __future__ import annotations

import os
import json
import shlex
import sqlite3
import sys
import tempfile

import pytest

os.environ.setdefault("LIBAGENTS_HOME", tempfile.mkdtemp(prefix="libagents-test-"))

from libagents import control, environment, paths  # noqa: E402
from libagents.board import SCHEMA as BOARD_SCHEMA  # noqa: E402
from libagents.board import USER, Board, parse_target  # noqa: E402
from libagents.models import EnvConfig, RunnerConfig, UsageRow  # noqa: E402
from libagents.paths import PathMapper  # noqa: E402
from libagents.providers.openrouter_provider import trim_to_valid_prefix  # noqa: E402
from libagents.pricing import cost as estimate_cost  # noqa: E402
from libagents.tools.base import compress  # noqa: E402
from libagents.llm import usage_from_response  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBAGENTS_HOME", str(tmp_path))
    name = "testenv"
    environment.create_environment(name, EnvConfig(sandbox="local"))
    environment.create_profile(name, "alice", RunnerConfig(goal="test"))
    environment.create_profile(name, "bob", RunnerConfig(goal="test"))
    return name


# ------------------------------------------------------------------- paths


def test_path_mapper_blocks_escapes(tmp_path):
    root = tmp_path / "env"
    (root / "agents").mkdir(parents=True)
    m = PathMapper(root, "/env")
    assert m.to_host("/env/agents") == (root / "agents").resolve()
    assert m.to_host("agents") == (root / "agents").resolve()
    assert m.to_host("/agents") == (root / "agents").resolve()
    for bad in ["../etc/passwd", "/env/../../etc", "agents/../../.."]:
        with pytest.raises(ValueError):
            m.to_host(bad)


# ------------------------------------------------------------------- board


def test_parse_target():
    assert parse_target("#general") == ("general", None)
    assert parse_target("@alice") == (None, "alice")
    assert parse_target("alice") == (None, "alice")
    with pytest.raises(ValueError):
        parse_target("")


def test_unread_excludes_own_and_advances_cursor(env):
    b = Board(paths.board_db(env))
    b.send("alice", "#general", "hi all")
    b.send(USER, "@bob", "ping")
    assert b.unread_count("alice") == 0, "own channel message must not be unread to the sender"
    assert b.unread_count("bob") == 2
    got = b.fetch_unread("bob")
    assert [m.body for m in got] == ["hi all", "ping"]
    assert b.unread_count("bob") == 0
    # read_history must not move the cursor
    b.send("alice", "@bob", "again")
    b.history("bob", "@alice")
    assert b.unread_count("bob") == 1


def test_unread_after_baseline_prevents_wake_spin(env):
    """An agent that sleeps without clearing its inbox must not be woken by
    the messages it already ignored."""
    b = Board(paths.board_db(env))
    b.send("alice", "@bob", "old")
    baseline = b.max_id()
    assert b.unread_count("bob", after=baseline) == 0
    b.send("alice", "@bob", "new")
    assert b.unread_count("bob", after=baseline) == 1


def test_scope_read_does_not_acknowledge_another_scope(env):
    b = Board(paths.board_db(env))
    b.ensure_channel("private")
    b.subscribe("bob", "private")
    b.send("alice", "#general", "public update")
    b.send("alice", "#private", "private update")

    public, more = b.fetch_unread_scope("bob", "#general")
    assert [m.body for m in public] == ["public update"]
    assert not more
    assert b.unread_count("bob") == 1
    assert [m.body for m in b.fetch_unread("bob")] == ["private update"]


def test_joining_channel_starts_at_current_history(env):
    b = Board(paths.board_db(env))
    b.send("alice", "#research", "before bob joined")
    b.subscribe("bob", "research")
    assert b.fetch_unread("bob") == []
    b.send("alice", "#research", "after bob joined")
    assert [m.body for m in b.fetch_unread("bob")] == ["after bob joined"]


def test_legacy_global_cursor_is_migrated_per_scope(tmp_path):
    db = tmp_path / "board.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(BOARD_SCHEMA)
        conn.execute("INSERT INTO subscriptions VALUES ('bob', 'general')")
        conn.execute(
            "INSERT INTO messages (ts,sender,channel,recipient,body) VALUES (?,?,?,?,?)",
            ("now", "alice", "general", None, "already read"),
        )
        conn.execute(
            "INSERT INTO messages (ts,sender,channel,recipient,body) VALUES (?,?,?,?,?)",
            ("now", "alice", "general", None, "still unread"),
        )
        conn.execute("INSERT INTO cursors VALUES ('bob', 1)")

    assert [m.body for m in Board(db).fetch_unread("bob")] == ["still unread"]


def test_long_message_spills_to_file(env, tmp_path):
    b = Board(paths.board_db(env))
    spill = paths.shared_dir(env) / "messages"
    mid, truncated = b.send("alice", "@bob", "x" * 9000, max_chars=100, spill_dir=spill)
    assert truncated
    msg = b.fetch_unread("bob")[0]
    assert msg.spill_path and not os.path.isabs(msg.spill_path)
    assert len((paths.env_dir(env) / msg.spill_path).read_text()) == 9000
    docker_mapper = PathMapper(paths.env_dir(env), "/env")
    assert docker_mapper.to_host(msg.spill_path) == (paths.env_dir(env) / msg.spill_path).resolve()
    assert len(msg.body) < 300


def test_channel_subscribers_are_explicit(env):
    b = Board(paths.board_db(env))
    b.ensure_channel("private")
    b.subscribe("alice", "private")
    assert b.subscribers("private") == {"alice"}
    assert b.unsubscribe("alice", "private")
    assert not b.unsubscribe("alice", "private")
    assert b.subscribers("private") == set()


def test_general_is_the_only_default_channel(env):
    b = Board(paths.board_db(env))
    assert [channel["name"] for channel in b.list_channels()] == ["general"]
    assert b.subscribers("general") == {USER, "alice", "bob"}


def test_leave_channel_tool_stops_channel_delivery(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    b = Board(paths.board_db(env))
    b.ensure_channel("private")
    b.subscribe("alice", "private")
    runner = Runner(env, "alice")
    result = runner._execute(ToolCall("1", "leave_channel", {"channel": "#private"}))
    assert "left #private" in result
    b.send("bob", "#private", "secret")
    assert b.fetch_unread("alice") == []


# ------------------------------------------------------------------- tools


def test_send_message_blocks_on_unread_scope_then_retries(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    board = Board(paths.board_db(env))
    for i in range(6):
        board.send("bob", "#general", f"new context {i}")
    runner = Runner(env, "alice")

    blocked = runner._execute(
        ToolCall("1", "send_message", {"to": "#general", "body": "stale reply"})
    )
    assert "MESSAGE NOT SENT" in blocked
    assert "new context 0" in blocked and "new context 4" in blocked
    assert "new context 5" not in blocked
    assert "More unread messages remain" in blocked
    assert board.unread_count("alice") == 1
    assert "stale reply" not in [m.body for m in board.recent()]

    blocked_again = runner._execute(
        ToolCall("2", "send_message", {"to": "#general", "body": "updated reply"})
    )
    assert "MESSAGE NOT SENT" in blocked_again and "new context 5" in blocked_again

    sent = runner._execute(
        ToolCall("3", "send_message", {"to": "#general", "body": "updated reply"})
    )
    assert "sent #" in sent
    assert [m.body for m in board.recent()][-1] == "updated reply"


def test_send_anyway_escapes_continuously_busy_scope(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    board = Board(paths.board_db(env))
    board.send("bob", "@alice", "another update")
    runner = Runner(env, "alice")
    sent = runner._execute(
        ToolCall(
            "1",
            "send_message",
            {"to": "@bob", "body": "still relevant", "send_anyway": True},
        )
    )
    assert "sent #" in sent
    assert [m.body for m in board.history("alice", "@bob")][-1] == "still relevant"


def test_wake_injects_bounded_unread_message_page(env):
    from libagents.runner import Runner

    board = Board(paths.board_db(env))
    for i in range(6):
        board.send("bob", "@alice", f"wake message {i}")

    item = Runner(env, "alice")._wake_item("new message")
    text = item["content"][0]["text"]
    assert "MESSAGES DELIVERED ON WAKE" in text
    assert "wake message 0" in text and "wake message 4" in text
    assert "wake message 5" not in text
    assert "1 unread message(s) remain" in text
    assert board.unread_count("alice") == 1


def test_default_memory_is_only_the_state_snapshot_hint():
    assert environment.DEFAULT_MEMORY == (
        "# memory.md\n\n"
        "*Your state snapshot. Survives compaction; nothing else does.*\n"
    )


def test_memory_guard_rejects_oversize_write(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    control.upsert_runner(env, "alice", RunnerConfig(memory_char_limit=100))
    r = Runner(env, "alice")
    before = paths.memory_file(env, "alice").read_text()
    out = r._execute(ToolCall("1", "write_file", {"path": "/agents/alice/memory.md", "content": "y" * 500}))
    assert "ERROR" in out and "limit is 100" in out
    assert paths.memory_file(env, "alice").read_text() == before, "failed write must not touch the file"
    ok = r._execute(ToolCall("2", "write_file", {"path": "/agents/alice/memory.md", "content": "small"}))
    assert "ERROR" not in ok


def test_read_file_cap_returns_exact_continuation(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    control.upsert_runner(env, "alice", RunnerConfig(read_file_char_limit=7))
    target = paths.shared_dir(env) / "long.txt"
    target.write_text("AAAA\nBBBB\nCCCC\n", encoding="utf-8")
    runner = Runner(env, "alice")

    first = runner._execute(
        ToolCall("1", "read_file", {"path": "/shared/long.txt", "start_line": 1, "num_lines": 3})
    )
    assert "AAAA\nBB" in first
    assert '"start_char": 7' in first

    second = runner._execute(
        ToolCall("2", "read_file", {"path": "/shared/long.txt", "start_char": 7})
    )
    assert "BB\nCCC" in second
    assert '"start_char": 14' in second


def test_status_line_is_on_every_tool_result(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    r = Runner(env, "alice")
    for call in [ToolCall("1", "list_agents", {}), ToolCall("2", "nope", {})]:
        assert r._execute(call).startswith("STATUS tokens_in=")


def test_agent_instructions_are_frozen_for_a_run(env):
    from libagents.runner import Runner

    r = Runner(env, "alice")
    original = r.instructions()
    r._instructions_snapshot = original
    r.config.base_prompt_override = "changed during this run"
    assert r._instructions_for_run() == original
    assert "changed during this run" in r.instructions()


def test_shell_session_persists_cwd_and_exports(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner
    from libagents.sandbox import close_shell_sessions

    r = Runner(env, "alice")
    try:
        r._execute(ToolCall("1", "shell", {"command": "cd ../../shared"}))
        pwd = r._execute(ToolCall("2", "shell", {"command": "pwd"}))
        assert str(paths.shared_dir(env)) in pwd
        r._execute(ToolCall("3", "shell", {"command": "export PERSISTED=yes"}))
        exported = r._execute(ToolCall("4", "shell", {"command": "printf \"$PERSISTED\""}))
        assert exported.endswith("yes")
    finally:
        close_shell_sessions(env, "alice")


def test_sleep_persists_a_message_boundary(env):
    from libagents.runner import Runner
    from libagents.tools.base import Sleep

    board = Board(paths.board_db(env))
    board.send("alice", "@bob", "old")
    r = Runner(env, "bob")
    assert r._sleep(Sleep(None, "waiting")) == "waiting"
    parked = control.get_runner(env, "bob")
    assert parked.wake_after_id == board.max_id()
    board.send("alice", "@bob", "new")
    assert board.unread_count("bob", after=parked.wake_after_id) == 1


def test_compress_keeps_head_and_tail():
    text = "\n".join(str(i) for i in range(100))
    out, elided = compress(text, 3, 3, 100)
    assert elided
    assert out.startswith("0\n1\n2\n") and out.endswith("97\n98\n99")
    short, elided = compress("a\nb", 3, 3, 100)
    assert short == "a\nb" and not elided
    clipped, elided = compress("x" * 500, 3, 3, 10)
    assert clipped.startswith("xxxxxxxxxx ...[clipped]")
    assert elided, "clipped single-line output must be spilled by the shell tool"


def test_duplicate_profile_is_rejected_and_delete_cleans_state(env):
    control.record_usage(env, "alice", "test", UsageRow(input_tokens=7))
    with pytest.raises(ValueError):
        environment.create_profile(env, "alice", RunnerConfig(model="other"))
    environment.delete_profile(env, "alice")
    assert not control.runner_exists(env, "alice")
    assert control.usage_for(env, "alice").input_tokens == 0
    assert "alice" not in {a["name"] for a in Board(paths.board_db(env)).list_agents()}


def test_complete_agent_reset_keeps_config_but_erases_agent_state(env):
    from libagents import api

    original = RunnerConfig(goal="keep this goal", model="keep-this-model")
    control.upsert_runner(env, "alice", original)
    private_file = paths.profile_dir(env, "alice") / "private.txt"
    private_file.write_text("erase me", encoding="utf-8")
    paths.memory_file(env, "alice").write_text("old memory", encoding="utf-8")
    control.record_usage(env, "alice", original.model, UsageRow(input_tokens=123))
    board = Board(paths.board_db(env))
    board.send("alice", "#general", "erase sent channel message")
    board.send(USER, "@alice", "erase received dm")
    board.send("bob", "#general", "keep other agent message")

    result = api.agent_action(env, "alice", "reset-complete")

    reset = control.get_runner(env, "alice")
    assert result["state"] == "inactive"
    assert reset.config == original
    assert control.usage_for(env, "alice").input_tokens == 0
    assert not private_file.exists()
    assert paths.memory_file(env, "alice").read_text() == environment.DEFAULT_MEMORY
    assert [m.body for m in board.recent()] == ["keep other agent message"]
    assert "alice" in board.subscribers("general")


def test_complete_environment_reset_recreates_all_agents(env):
    from libagents import api

    original = control.get_env(env)
    original_runners = {r.profile: r.config for r in control.list_runners(env)}
    environment.write_env_file(env, "SECRET=erase\n")
    doomed = paths.shared_dir(env) / "doomed.txt"
    doomed.write_text("erase", encoding="utf-8")
    control.record_usage(env, "alice", "test", UsageRow(input_tokens=123))

    result = api.environment_action(env, "reset-complete")

    assert result == {
        "ok": True,
        "reset": env,
        "reset_agents": ["alice", "bob"],
        "sandbox_running": False,
    }
    assert control.get_env(env) == original
    reset_runners = {r.profile: r.config for r in control.list_runners(env)}
    assert reset_runners == original_runners
    assert all(
        paths.memory_file(env, profile).read_text() == environment.DEFAULT_MEMORY
        for profile in reset_runners
    )
    assert control.usage_for(env, "alice").input_tokens == 0
    assert not doomed.exists()
    assert environment.read_env_file(env) == environment.DEFAULT_ENV_FILE
    board = Board(paths.board_db(env))
    assert {a["name"] for a in board.list_agents()} == {USER, "alice", "bob"}
    assert board.subscribers("general") == {USER, "alice", "bob"}


def test_delete_rejects_traversal_names(env, tmp_path):
    with pytest.raises(ValueError):
        environment.delete_environment("..")
    with pytest.raises(ValueError):
        environment.delete_profile(env, "..")
    assert paths.home().exists() and paths.env_dir(env).exists()


def test_spa_candidate_stays_inside_ui_dist(tmp_path, monkeypatch):
    from libagents import api

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "asset.js").write_text("ok")
    outside = tmp_path / "secret"
    outside.write_text("no")
    monkeypatch.setattr(api, "UI_DIST", dist)
    assert api._safe_ui_candidate("asset.js") == (dist / "asset.js").resolve()
    assert api._safe_ui_candidate("../secret") is None


def test_sandbox_start_errors_are_returned_as_service_unavailable(env, monkeypatch):
    from fastapi import HTTPException
    from libagents import api

    class BrokenSandbox:
        env_root = "/env"

        def start(self):
            raise RuntimeError("docker not found on PATH")

    monkeypatch.setattr(api, "make_sandbox", lambda *_: BrokenSandbox())
    monkeypatch.setattr(api.control, "list_runners", lambda *_: [])

    with pytest.raises(HTTPException) as action_error:
        api.environment_action(env, "start")
    assert action_error.value.status_code == 503
    assert "docker not found" in action_error.value.detail

    with pytest.raises(HTTPException) as console_error:
        api.exec_command(env, {"command": "pwd"})
    assert console_error.value.status_code == 503
    assert "docker not found" in console_error.value.detail


def test_environment_start_explicitly_starts_the_sandbox(env, monkeypatch):
    from libagents import api

    calls = []

    class FakeSandbox:
        env_root = "/env"

        def start(self):
            calls.append("start")

    monkeypatch.setattr(api, "make_sandbox", lambda *_: FakeSandbox())
    monkeypatch.setattr(api.control, "list_runners", lambda *_: [])

    result = api.environment_action(env, "start")
    assert calls == ["start"]
    assert result == {
        "ok": True, "sandbox_running": True, "started": [], "blocked": {}
    }


def test_openrouter_usage_details_and_reported_cost():
    class Details:
        cached_tokens = 80
        reasoning_tokens = 10

    class Usage:
        prompt_tokens = 100
        completion_tokens = 20
        prompt_tokens_details = Details()
        completion_tokens_details = Details()
        cost = 0.0123

    parsed = usage_from_response("openrouter/model", Usage())
    assert parsed.cached_input_tokens == 80
    assert parsed.reasoning_tokens == 10
    assert parsed.cost_usd == pytest.approx(0.0123)
    assert parsed.cost_known


def test_unknown_model_pricing_is_not_reported_as_free():
    class Usage:
        prompt_tokens = 100
        completion_tokens = 20

    parsed = usage_from_response("unknown/model", Usage())
    assert parsed.cost_usd == 0
    assert not parsed.cost_known


@pytest.mark.parametrize(
    ("model", "short_input", "long_input", "long_output"),
    [
        ("gpt-5.6-sol", 4.00, 8.00, 30.00),
        ("gpt-5.6-terra", 2.00, 4.00, 18.00),
        ("gpt-5.6-luna", 0.20, 0.40, 1.80),
    ],
)
def test_gpt56_short_and_long_context_pricing(model, short_input, long_input, long_output):
    assert estimate_cost(model, 100_000, 0, 0) == pytest.approx(short_input * 0.1)
    assert estimate_cost(model, 300_000, 0, 100_000) == pytest.approx(
        long_input * 0.3 + long_output * 0.1
    )


def test_gpt56_pricing_counts_cached_reads_and_cache_writes_separately():
    # 50K fresh + 20K cached + 30K cache writes + 10K output at Sol short rates.
    assert estimate_cost("gpt-5.6-sol", 100_000, 20_000, 10_000, 30_000) == pytest.approx(
        0.05 * 4.00 + 0.02 * 0.40 + 0.03 * 5.00 + 0.01 * 20.00
    )

    class Details:
        cached_tokens = 20_000
        cache_write_tokens = 30_000

    class Usage:
        input_tokens = 100_000
        output_tokens = 10_000
        input_tokens_details = Details()

    parsed = usage_from_response("gpt-5.6-sol", Usage())
    assert parsed.cache_write_tokens == 30_000
    assert parsed.cost_usd == pytest.approx(0.558)
    assert parsed.cost_known


def test_existing_unknown_gpt56_usage_is_backfilled():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(control.SCHEMA)
        conn.execute(
            """INSERT INTO usage (
                 env,profile,ts,model,input_tokens,cached_input_tokens,
                 cache_write_tokens,output_tokens,reasoning_tokens,cost_usd,cost_known
               ) VALUES ('demo','alice','now','gpt-5.6-luna',100000,0,0,0,0,0,0)"""
        )
        control._backfill_gpt56_pricing(conn)
        row = conn.execute("SELECT cost_usd, cost_known FROM usage").fetchone()
        assert row["cost_usd"] == pytest.approx(0.02)
        assert row["cost_known"] == 1
    finally:
        conn.close()


def test_openrouter_web_search_uses_server_tool(monkeypatch):
    import httpx
    from openai import OpenAI
    from libagents.providers import openrouter_provider
    from libagents.tools.base import specs_for

    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek/deepseek-v4-flash-0731",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "result"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0.001,
                    "server_tool_use": {"web_search_requests": 1},
                },
            },
        )

    client = OpenAI(
        api_key="test",
        base_url="https://openrouter.invalid/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(openrouter_provider, "openrouter_client", lambda: client)
    provider = openrouter_provider.OpenRouterProvider(
        RunnerConfig(provider="openrouter", model="deepseek/deepseek-v4-flash-0731")
    )
    turn = provider.generate(
        instructions="system",
        items=[provider.user_item("current news")],
        tools=specs_for(["web_search"]),
        max_output_tokens=123,
    )
    assert turn.text == "result" and turn.usage.output_tokens == 2
    assert captured["tools"][0]["type"] == "openrouter:web_search"
    assert captured["tools"][0]["parameters"]["engine"] == "auto"
    assert captured["tools"][0]["parameters"]["max_total_results"] == 10
    assert captured["max_tool_calls"] == 3
    assert captured["max_tokens"] == 123


# --------------------------------------------------------------- providers


def test_trim_to_valid_prefix_keeps_tool_calls_paired():
    items = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function", "function": {}}]},
    ]
    out = trim_to_valid_prefix(items)
    assert len(out) == 3, "the trailing unanswered tool call must be dropped"
    assert trim_to_valid_prefix([{"role": "tool", "tool_call_id": "z", "content": "orphan"}]) == []


# --------------------------------------------------------- control vs sandbox


def test_runner_config_lives_outside_the_environment(env):
    """The budget must not be reachable from inside the sandbox."""
    files = [p.name for p in paths.profile_dir(env, "alice").rglob("*")]
    assert "runner.json" not in files
    blob = "".join(
        p.read_text(errors="ignore")
        for p in paths.env_dir(env).rglob("*")
        if p.is_file() and p.suffix in {".md", ".json"}
    )
    assert "input_tokens" not in blob
    assert control.get_runner(env, "alice").config.budgets.input_tokens > 0


def test_legacy_environment_goal_and_summary_settings_migrate_to_runners(env):
    db = sqlite3.connect(paths.control_db())
    try:
        env_cfg = json.loads(
            db.execute("SELECT config FROM environments WHERE name=?", (env,)).fetchone()[0]
        )
        env_cfg.update({
            "goal": "legacy goal",
            "summary_provider": "openai",
            "summary_model": "legacy-summary-model",
        })
        runner_cfg = json.loads(
            db.execute(
                "SELECT config FROM runners WHERE env=? AND profile='alice'", (env,)
            ).fetchone()[0]
        )
        for key in ("goal", "summary_provider", "summary_model"):
            runner_cfg.pop(key, None)
        db.execute(
            "UPDATE environments SET config=? WHERE name=?", (json.dumps(env_cfg), env)
        )
        db.execute(
            "UPDATE runners SET config=? WHERE env=? AND profile='alice'",
            (json.dumps(runner_cfg), env),
        )
        db.commit()
    finally:
        db.close()

    migrated = control.get_runner(env, "alice").config
    assert migrated.goal == "legacy goal"
    assert migrated.summary_provider == "openai"
    assert migrated.summary_model == "legacy-summary-model"
    assert "goal" not in control.list_envs()[0]["config"]


def test_legacy_runner_prompt_and_tools_migrate_once(env):
    db = sqlite3.connect(paths.control_db())
    try:
        raw = db.execute(
            "SELECT config FROM runners WHERE env=? AND profile='alice'", (env,)
        ).fetchone()[0]
        config = json.loads(raw)
        config["operator_instructions"] = "retired"
        config["tools"].remove("leave_channel")
        db.execute(
            "UPDATE runners SET config=? WHERE env=? AND profile='alice'",
            (json.dumps(config), env),
        )
        db.execute(
            "DELETE FROM schema_migrations WHERE name='runner-prompt-and-leave-channel-v1'"
        )
        db.commit()
    finally:
        db.close()

    migrated = control.get_runner(env, "alice").config
    assert "leave_channel" in migrated.tools
    assert "operator_instructions" not in migrated.model_dump()

    migrated.tools.remove("leave_channel")
    control.upsert_runner(env, "alice", migrated)
    assert "leave_channel" not in control.get_runner(env, "alice").config.tools


def test_environment_dotenv_and_forwarded_host_values(env, monkeypatch):
    from libagents.sandbox import _sandbox_env

    environment.write_env_file(env, "LOCAL_VALUE=inside\nCOLLISION=file\n")
    monkeypatch.setenv("HOST_VALUE", "forwarded")
    monkeypatch.setenv("COLLISION", "host")
    cfg = EnvConfig(sandbox="local", secrets=["HOST_VALUE", "COLLISION"])
    values = _sandbox_env(env, cfg)
    assert values == {
        "LOCAL_VALUE": "inside",
        "COLLISION": "host",
        "HOST_VALUE": "forwarded",
    }


def test_macos_sandbox_uses_canonical_root_and_environment_local_homes(
    env, monkeypatch
):
    from libagents.sandbox import MacOSSandbox, _macos_host_env

    environment.write_env_file(env, "FILE_VALUE=yes\nHOME=/must-not-win\n")
    monkeypatch.setenv("FORWARDED_VALUE", "available")
    monkeypatch.setenv("UNFORWARDED_SECRET", "hidden")
    cfg = EnvConfig(sandbox="macos", secrets=["FORWARDED_VALUE"])
    sandbox = MacOSSandbox(env, cfg)
    environ = _macos_host_env(env, cfg, sandbox.host_root)

    assert sandbox.env_root == str(paths.env_dir(env).resolve())
    assert environ["FILE_VALUE"] == "yes"
    assert environ["FORWARDED_VALUE"] == "available"
    assert "UNFORWARDED_SECRET" not in environ
    assert environ["HOME"].startswith(f"{sandbox.env_root}/.sandbox/")
    assert environ["TMPDIR"].startswith(f"{sandbox.env_root}/.sandbox/")
    assert environ["CFFIXED_USER_HOME"] == environ["HOME"]
    assert environ["CLANG_MODULE_CACHE_PATH"].startswith(
        f"{sandbox.env_root}/.sandbox/cache/"
    )


def test_host_sandbox_status_marker_tracks_start_and_stop(env):
    from libagents.sandbox import LocalSandbox

    sandbox = LocalSandbox(env, EnvConfig(sandbox="local"))
    assert not sandbox.running()
    sandbox.start()
    assert sandbox.running()
    sandbox.stop()
    assert not sandbox.running()


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("LIBAGENTS_TEST_MACOS_SANDBOX") != "1",
    reason="opt-in macOS Seatbelt integration test",
)
def test_macos_sandbox_confines_writes_and_preserves_shell_sessions(env):
    from libagents.sandbox import MacOSSandbox

    cfg = EnvConfig(sandbox="macos")
    sandbox = MacOSSandbox(env, cfg)
    outside = paths.home() / "outside-seatbelt-canary"
    outside.unlink(missing_ok=True)
    try:
        first = sandbox.exec(
            "cd ../../shared; printf inside > seatbelt-canary",
            cwd=str(paths.profile_dir(env, "alice")),
            timeout=10,
            session="alice",
        )
        assert first.exit_code == 0
        second = sandbox.exec("pwd", cwd=sandbox.env_root, timeout=10, session="alice")
        assert second.output == str(paths.shared_dir(env).resolve())
        denied = sandbox.exec(
            f"printf outside > {shlex.quote(str(outside))}",
            cwd=sandbox.env_root,
            timeout=10,
        )
        assert denied.exit_code != 0
        assert not outside.exists()
        via_symlink = sandbox.exec(
            f"ln -sf {shlex.quote(str(outside))} escape-canary; "
            "printf outside > escape-canary",
            cwd=sandbox.env_root,
            timeout=10,
        )
        assert via_symlink.exit_code != 0
        assert not outside.exists()
    finally:
        sandbox.stop()
        outside.unlink(missing_ok=True)


def test_agent_detail_exposes_exact_prompt_and_tool_payloads(env):
    from libagents import api

    cfg = RunnerConfig(
        base_prompt_override="Custom operator base.",
        goal="Agent-specific goal.",
        tools=["web_search", "sleep"],
        tool_description_overrides={"sleep": "Custom sleep description."},
    )
    control.upsert_runner(env, "alice", cfg)
    detail = api.get_agent(env, "alice")
    assert detail["prompt"]["system_prompt"] == "Custom operator base."
    assert "Agent-specific goal." in detail["prompt"]["injected"][0]["content"]
    tools = {t["name"]: t for t in detail["prompt"]["tools"]}
    assert tools["web_search"]["provider_payload"] == {"type": "web_search"}
    assert tools["sleep"]["provider_payload"]["description"] == "Custom sleep description."


def test_default_prompt_encourages_collaboration_and_token_efficiency():
    from libagents import prompts

    prompt = prompts.instructions("alice", "/env", 6000)
    assert "Other agents may have related or overlapping goals" in prompt
    assert "Check who else is active and coordinate" in prompt
    assert "start subscribed only to `#general`" in prompt
    assert "do not receive their messages until you do" in prompt
    assert "Aim to finish\nwithin your assigned budget" in prompt
    assert "Aim to be token efficient" in prompt


# -------------------------------------------------------------- compaction


def _runner(env, **overrides):
    from libagents.runner import Runner

    control.upsert_runner(env, "alice", RunnerConfig(**overrides))
    return Runner(env, "alice")


def test_model_swap_is_summarized_by_the_previous_model(env, monkeypatch):
    from libagents import runner as runner_module
    from libagents.runner import Conversation

    r = _runner(env, model="new-model")
    encrypted = {"type": "reasoning", "encrypted_content": "opaque"}
    Conversation(
        provider="openai", model="old-model", items=[encrypted], last_input_tokens=10
    ).save(r.conv_path)
    seen = {}

    class OldProvider:
        def __init__(self, cfg):
            self.model = cfg.model

        def summarize(self, **kwargs):
            seen["model"] = self.model
            seen["items"] = kwargs["items"]
            return "summary from old model", UsageRow(input_tokens=1)

    monkeypatch.setattr(runner_module, "make_provider", lambda cfg: OldProvider(cfg))
    conv = r._load_conversation()
    assert seen == {"model": "old-model", "items": [encrypted]}
    assert conv.summary == "summary from old model"


def test_native_compaction_cuts_at_the_checkpoint(env):
    from libagents.providers.base import Turn
    from libagents.runner import Conversation

    r = _runner(env)
    checkpoint = {"type": "compaction", "encrypted_content": "zz"}
    turn = Turn(items=[checkpoint, {"type": "message"}], compaction_items=[checkpoint])
    conv = Conversation(items=[{"role": "user", "content": "old"}] * 4 + turn.items)

    r._apply_native_compaction(conv, turn)

    assert conv.items[0] is checkpoint, "everything before the checkpoint must be dropped"
    assert conv.compactions == 1 and conv.last_input_tokens == 0
    # The prefix is re-added so memory.md survives the boundary.
    tail = " ".join(str(i) for i in conv.items[-2:])
    assert "YOUR GOAL" in tail and "memory.md" in tail


def test_native_compaction_uses_the_latest_checkpoint(env):
    from libagents.providers.base import Turn
    from libagents.runner import Conversation

    r = _runner(env)
    first = {"type": "compaction", "encrypted_content": "first"}
    latest = {"type": "compaction", "encrypted_content": "latest"}
    turn = Turn(items=[first, {"type": "message"}, latest], compaction_items=[first, latest])
    conv = Conversation(items=[{"role": "user", "content": "old"}] + turn.items)

    r._apply_native_compaction(conv, turn)

    assert conv.items[0] is latest
    assert first not in conv.items


def test_native_compaction_refuses_to_orphan_a_tool_call(env):
    """A checkpoint emitted after a function_call would leave its result
    stranded, which the API rejects. Skip the cut instead."""
    from libagents.providers.base import Turn
    from libagents.runner import Conversation

    r = _runner(env)
    checkpoint = {"type": "compaction", "encrypted_content": "zz"}
    turn = Turn(
        items=[{"type": "function_call", "call_id": "c1", "name": "shell"}, checkpoint],
        compaction_items=[checkpoint],
    )
    conv = Conversation(items=[{"role": "user", "content": "old"}] + turn.items)
    r._apply_native_compaction(conv, turn)
    assert conv.compactions == 0
    assert conv.items[0] == {"role": "user", "content": "old"}


def test_manual_compaction_only_backstops_the_native_path(env):
    from libagents.runner import Conversation

    native = _runner(env, compact_at_input_tokens=1000)
    assert native._native_compaction_active()
    assert not native._needs_manual_compaction(Conversation(last_input_tokens=1500))
    assert native._needs_manual_compaction(Conversation(last_input_tokens=2500)), "backstop"

    manual = _runner(env, compact_at_input_tokens=1000)
    manual.provider.native_compaction = False
    assert not manual._native_compaction_active()
    assert manual._needs_manual_compaction(Conversation(last_input_tokens=1500))


def test_openai_provider_sends_context_management():
    from libagents.providers.openai_provider import OpenAIProvider
    from libagents.tools.base import specs_for

    p = OpenAIProvider(RunnerConfig(compact_at_input_tokens=50_000))
    kwargs = p._kwargs("sys", [], [])
    assert kwargs["context_management"] == [{"type": "compaction", "compact_threshold": 50_000}]
    # The API floor is respected, and a rejection disables it cleanly.
    low = OpenAIProvider(RunnerConfig(compact_at_input_tokens=10))
    assert low._kwargs("sys", [], [])["context_management"][0]["compact_threshold"] == 1000
    p._disabled.add("context_management")
    assert not p.uses_native_compaction and "context_management" not in p._kwargs("sys", [], [])
    assert p._kwargs("sys", [], [], max_output_tokens=321)["max_output_tokens"] == 321
    tools = p._kwargs("sys", [], specs_for(["web_search", "sleep"]))["tools"]
    assert tools[0] == {"type": "web_search"}
    assert tools[1]["type"] == "function" and tools[1]["name"] == "sleep"


def test_exhausted_runner_cannot_get_fresh_grace_by_restarting(env):
    from libagents.supervisor import BudgetLimitError, SUPERVISOR

    control.upsert_runner(
        env, "alice", RunnerConfig(budgets={"input_tokens": 10, "output_tokens": 100})
    )
    control.record_usage(env, "alice", "test", UsageRow(input_tokens=10))
    with pytest.raises(BudgetLimitError, match="raise the runner limit"):
        SUPERVISOR.start(env, "alice")


def test_input_budget_increment_is_additive_and_live_runner_refreshes_it(env):
    from libagents.runner import Runner

    control.upsert_runner(
        env, "alice", RunnerConfig(
            goal="preserved",
            budgets={"input_tokens": 100, "output_tokens": 200},
        )
    )
    live = Runner(env, "alice")
    updated = control.add_input_budget(env, "alice", 500_000)

    assert updated.goal == "preserved"
    assert updated.budgets.input_tokens == 500_100
    assert updated.budgets.output_tokens == 200
    assert live.config.budgets.input_tokens == 100
    assert live._over_budget() is None
    assert live.config.budgets.input_tokens == 500_100


def test_budget_api_adds_tokens_and_resumes_finished_agent(env, monkeypatch):
    from libagents import api

    control.set_state(env, "alice", "finished", stop_reason="input token budget exhausted")
    started = []
    monkeypatch.setattr(api.SUPERVISOR, "is_running", lambda *_: False)
    monkeypatch.setattr(
        api.SUPERVISOR, "start", lambda e, p, reason: started.append((e, p, reason))
    )

    result = api.add_agent_budget(
        env, "alice", api.InputBudgetAdjustment(input_tokens=500_000)
    )

    assert result["input_budget"] == 1_500_000
    assert result["resumed"] and result["resume_error"] is None
    assert started == [(env, "alice", "input budget increased by operator")]


def test_budget_api_adjusts_active_runner_without_stopping_it(env, monkeypatch):
    from libagents import api

    control.set_state(env, "alice", "active")
    monkeypatch.setattr(api.SUPERVISOR, "is_running", lambda *_: True)
    monkeypatch.setattr(
        api.SUPERVISOR, "start",
        lambda *_: pytest.fail("an active runner must not be started again"),
    )
    monkeypatch.setattr(
        api.SUPERVISOR, "wake",
        lambda *_: pytest.fail("an active runner does not need a wake signal"),
    )

    result = api.add_agent_budget(
        env, "alice", api.InputBudgetAdjustment(input_tokens=1_000_000)
    )

    assert result["resumed"]
    assert result["input_budget"] == 2_000_000
    assert control.get_runner(env, "alice").state == "active"


def test_input_budget_increment_rejects_non_positive_values(env):
    with pytest.raises(ValueError, match="positive"):
        control.add_input_budget(env, "alice", 0)
