"""The agent loop.

Context layout, in order, and why:

    instructions          static; the cached prefix (AGENT.md folded in)
    [user] PROJECT GOAL   static
    [user] COMPACTED CTX  rewritten only at a compaction
    [user] memory.md      snapshot taken at the last compaction, then frozen
    ... everything since the last compaction

Freezing the memory snapshot between compactions is what keeps prompt caching
working: the prefix only ever changes at a compaction boundary, and
everything else is appended. Volatile state (token budget, unread count)
rides in on the STATUS line of each tool result, which is append-only too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import control, paths, prompts
from .board import Board
from .events import EventLog
from .models import EnvConfig, RunnerConfig, UsageRow
from .paths import PathMapper
from .providers import ToolCall, make_provider
from .sandbox import make_sandbox
from .tools.base import AgentContext, Finish, Sleep, ToolError, ToolSpec, specs_for

GRACE_STEPS = 3
"""Turns granted after the budget is blown, so the agent can save memory.md."""

MAX_NUDGES = 3


@dataclass
class Conversation:
    provider: str = "openai"
    model: str = ""
    items: list[dict] = field(default_factory=list)
    summary: Optional[str] = None
    compactions: int = 0
    last_input_tokens: int = 0

    @classmethod
    def load(cls, path: Path) -> "Conversation":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class BudgetExhausted(Exception):
    pass


class Runner:
    def __init__(self, env: str, profile: str, *, stop_flag=None):
        self.env = env
        self.profile = profile
        self.stop_flag = stop_flag
        self.runner = control.get_runner(env, profile)
        self.config: RunnerConfig = self.runner.config
        self.env_config: EnvConfig = control.get_env(env)

        self.sandbox = make_sandbox(env, self.env_config)
        self.mapper = PathMapper(paths.env_dir(env), self.sandbox.env_root)
        self.board = Board(paths.board_db(env))
        self.events = EventLog(paths.history_dir(env, profile) / "events.jsonl")
        self.conv_path = paths.history_dir(env, profile) / "conversation.json"

        self.provider = make_provider(self.config)
        self.tools: list[ToolSpec] = specs_for(self.config.tools)
        self.used = control.usage_for(env, profile)
        self.ctx = AgentContext(
            env=env,
            profile=profile,
            config=self.config,
            env_config=self.env_config,
            sandbox=self.sandbox,
            board=self.board,
            mapper=self.mapper,
            events=self.events,
            profile_dir=paths.profile_dir(env, profile),
            used=self.used,
        )
        self._pending: Optional[Exception] = None
        self._grace = -1

    # ----------------------------------------------------------- prompt bits

    def instructions(self) -> str:
        am = paths.agent_md(self.env, self.profile)
        return prompts.instructions(
            self.profile,
            self.sandbox.env_root,
            self.config.memory_char_limit,
            am.read_text(encoding="utf-8") if am.exists() else "",
        )

    def _memory_text(self) -> str:
        mem = paths.memory_file(self.env, self.profile)
        text = mem.read_text(encoding="utf-8") if mem.exists() else "(empty)"
        limit = self.config.memory_char_limit
        if len(text) > limit:
            # The edit tool refuses oversized writes, but a shell command can
            # bypass it. Keep the loop alive and make the loss visible.
            text = text[:limit] + "\n[TRUNCATED -- memory.md exceeded its limit]"
        return text

    def _prefix(self, conv: Conversation) -> list[dict]:
        items = [self.provider.user_item(prompts.GOAL_BLOCK.format(goal=self.env_config.goal or "(none)"))]
        if conv.summary:
            items.append(self.provider.user_item(prompts.COMPACTION_BLOCK.format(summary=conv.summary)))
        items.append(self.provider.user_item(prompts.MEMORY_BLOCK.format(memory=self._memory_text())))
        return items

    # ------------------------------------------------------------- lifecycle

    def _load_conversation(self) -> Conversation:
        conv = Conversation.load(self.conv_path)
        fresh = not conv.items
        swapped = bool(conv.items) and (
            conv.provider != self.config.provider or conv.model != self.config.model
        )
        if swapped:
            # Encrypted reasoning is bound to the model that produced it, and
            # item shapes differ per provider. Compact, then start clean.
            self.events.emit(
                "model_swap", old=f"{conv.provider}/{conv.model}",
                new=f"{self.config.provider}/{self.config.model}",
            )
            self._compact(conv, reason="model swap")
        if fresh or swapped or self.config.memoryless:
            conv.items = self._prefix(conv)
        conv.provider, conv.model = self.config.provider, self.config.model
        return conv

    def _over_budget(self) -> Optional[str]:
        b = self.config.budgets
        if self.used.input_tokens >= b.input_tokens:
            return "input token budget exhausted"
        if self.used.output_tokens >= b.output_tokens:
            return "output token budget exhausted"
        cap = self.env_config.input_token_cap
        if cap and control.usage_for(self.env).input_tokens >= cap:
            return "environment-wide token cap reached"
        return None

    def run(self, wake_reason: str = "started") -> str:
        self.sandbox.start()
        self.board.register(self.profile, state="active")
        self.board.set_status(self.profile, state="active")
        control.set_state(self.env, self.profile, "active")
        self.events.emit("run_start", reason=wake_reason)

        conv = self._load_conversation()
        unread = self.board.unread_count(self.profile)
        conv.items.append(self.provider.user_item(prompts.WAKE.format(reason=wake_reason, unread=unread)))

        try:
            state = self._loop(conv)
        except BudgetExhausted as exc:
            state = self._stop("finished", str(exc))
        except Exception as exc:  # provider/network failure: park, don't lose context
            self.events.emit("error", message=f"{type(exc).__name__}: {exc}")
            state = self._stop("waiting", f"error: {type(exc).__name__}: {exc}")
        finally:
            conv.save(self.conv_path)
        return state

    def _loop(self, conv: Conversation) -> str:
        nudges = 0
        for step in range(self.config.max_steps_per_wake):
            if self.stop_flag is not None and self.stop_flag.is_set():
                return self._stop("waiting", "stopped by operator")

            reason = self._over_budget()
            if reason:
                if self._grace < 0:
                    self._grace = GRACE_STEPS
                    conv.items.append(self.provider.user_item(prompts.BUDGET_EXHAUSTED))
                    self.events.emit("budget_exhausted", reason=reason)
                elif self._grace == 0:
                    raise BudgetExhausted(reason)
                else:
                    self._grace -= 1

            if conv.last_input_tokens > self.config.compact_at_input_tokens:
                self._compact(conv, reason="context threshold")
                conv.items = self._prefix(conv) + [
                    self.provider.user_item("Context was compacted. Continue where you left off.")
                ]

            turn = self.provider.generate(
                instructions=self.instructions(), items=conv.items, tools=self.tools
            )
            self._record(turn.usage)
            conv.last_input_tokens = turn.usage.input_tokens
            conv.items.extend(turn.items)
            if turn.reasoning:
                self.events.emit("reasoning", text=turn.reasoning)
            if turn.text:
                self.events.emit("message", text=turn.text)

            if not turn.tool_calls:
                nudges += 1
                if nudges > MAX_NUDGES:
                    return self._stop("waiting", "stopped calling tools")
                conv.items.append(self.provider.user_item(prompts.NO_TOOL_CALL))
                conv.save(self.conv_path)
                continue
            nudges = 0

            for call in turn.tool_calls:
                output = self._execute(call)
                conv.items.append(self.provider.tool_result_item(call, output))
            conv.save(self.conv_path)

            if isinstance(self._pending, Sleep):
                return self._sleep(self._pending)
            if isinstance(self._pending, Finish):
                return self._stop("finished", self._pending.summary or "finished")

        return self._stop("waiting", "step limit for this wake reached")

    # ----------------------------------------------------------------- tools

    def _execute(self, call: ToolCall) -> str:
        status = self.ctx.status_line()
        if call.parse_error:
            return f"{status}\nERROR: {call.parse_error}"
        spec = next((t for t in self.tools if t.name == call.name), None)
        if spec is None:
            return f"{status}\nERROR: no such tool {call.name!r}"

        self.events.emit("tool_call", tool=call.name, arguments=call.arguments)
        try:
            result = spec.fn(self.ctx, call.arguments)
        except (Sleep, Finish) as ctrl:
            self._pending = ctrl
            body = "sleeping" if isinstance(ctrl, Sleep) else "finished"
            self.events.emit("tool_result", tool=call.name, summary=body)
            return f"{self.ctx.status_line()}\n{body}"
        except ToolError as exc:
            self.events.emit("tool_result", tool=call.name, summary=f"ERROR: {exc}", error=True)
            return f"{self.ctx.status_line()}\nERROR: {exc}"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self.events.emit("tool_result", tool=call.name, summary=f"ERROR: {msg}", error=True)
            return f"{self.ctx.status_line()}\nERROR: {msg}"

        self.events.emit("tool_result", tool=call.name, summary=result.summary)
        return f"{self.ctx.status_line()}\n{result.summary}"

    # ----------------------------------------------------------- bookkeeping

    def _record(self, usage: UsageRow) -> None:
        control.record_usage(self.env, self.profile, self.config.model, usage)
        self.used.input_tokens += usage.input_tokens
        self.used.cached_input_tokens += usage.cached_input_tokens
        self.used.output_tokens += usage.output_tokens
        self.used.reasoning_tokens += usage.reasoning_tokens
        self.used.cost_usd += usage.cost_usd
        self.events.emit("usage", **usage.model_dump())

    def _compact(self, conv: Conversation, reason: str) -> None:
        if not conv.items:
            return
        try:
            summary, usage = self.provider.summarize(
                instructions=self.instructions(),
                items=conv.items,
                prompt=prompts.COMPACTION_PROMPT,
            )
            self._record(usage)
        except Exception as exc:
            self.events.emit("error", message=f"compaction failed: {exc}")
            summary = conv.summary or "(compaction failed; context was dropped)"
        conv.summary = summary
        conv.compactions += 1
        conv.last_input_tokens = 0
        self.events.emit("compaction", reason=reason, summary=summary, n=conv.compactions)

    def _sleep(self, sleep: Sleep) -> str:
        wake_at = time.time() + sleep.seconds if sleep.seconds else None
        status = sleep.status or ("sleeping" if wake_at is None else f"sleeping {sleep.seconds:.0f}s")
        self.board.set_status(self.profile, status=status, state="waiting")
        control.set_state(self.env, self.profile, "waiting", wake_at=wake_at)
        self.events.emit("sleep", seconds=sleep.seconds, status=status)
        if self.config.memoryless:
            # Reset context so the next wake starts from goal + memory.md only.
            Conversation(provider=self.config.provider, model=self.config.model).save(self.conv_path)
        return "waiting"

    def _stop(self, state: str, reason: str) -> str:
        self.board.set_status(self.profile, status=reason[:200], state=state)
        control.set_state(self.env, self.profile, state, stop_reason=reason)
        self.events.emit("run_end", state=state, reason=reason)
        return state
