"""Tests for the parts where a silent bug would be expensive: path escapes,
unread bookkeeping, the memory.md limit, and the tool-call/result pairing that
providers reject if we get it wrong."""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("LIBAGENTS_HOME", tempfile.mkdtemp(prefix="libagents-test-"))

from libagents import control, environment, paths  # noqa: E402
from libagents.board import USER, Board, parse_target  # noqa: E402
from libagents.models import EnvConfig, RunnerConfig  # noqa: E402
from libagents.paths import PathMapper  # noqa: E402
from libagents.providers.openrouter_provider import trim_to_valid_prefix  # noqa: E402
from libagents.tools.base import compress  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBAGENTS_HOME", str(tmp_path))
    name = "testenv"
    environment.create_environment(name, EnvConfig(sandbox="local", goal="test"))
    environment.create_profile(name, "alice", RunnerConfig())
    environment.create_profile(name, "bob", RunnerConfig())
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


def test_long_message_spills_to_file(env, tmp_path):
    b = Board(paths.board_db(env))
    spill = paths.shared_dir(env) / "messages"
    mid, truncated = b.send("alice", "@bob", "x" * 9000, max_chars=100, spill_dir=spill)
    assert truncated
    msg = b.fetch_unread("bob")[0]
    assert msg.spill_path and len(open(msg.spill_path).read()) == 9000
    assert len(msg.body) < 300


# ------------------------------------------------------------------- tools


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


def test_status_line_is_on_every_tool_result(env):
    from libagents.providers.base import ToolCall
    from libagents.runner import Runner

    r = Runner(env, "alice")
    for call in [ToolCall("1", "list_agents", {}), ToolCall("2", "nope", {})]:
        assert r._execute(call).startswith("STATUS tokens_in=")


def test_compress_keeps_head_and_tail():
    text = "\n".join(str(i) for i in range(100))
    out, elided = compress(text, 3, 3, 100)
    assert elided
    assert out.startswith("0\n1\n2\n") and out.endswith("97\n98\n99")
    short, elided = compress("a\nb", 3, 3, 100)
    assert short == "a\nb" and not elided
    clipped, _ = compress("x" * 500, 3, 3, 10)
    assert clipped.startswith("xxxxxxxxxx ...[clipped]")


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
