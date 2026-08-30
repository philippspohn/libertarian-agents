"""HTTP API and static UI host.

`libagents serve` is also the daemon: sleeping agents live as threads in this
process, so this is what keeps a multi-agent environment alive between wakes.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import control, environment, paths, prompts
from .board import USER, Board
from .events import EventLog
from .models import EnvConfig, RunnerConfig
from .sandbox import close_shell_sessions, make_sandbox
from .supervisor import SUPERVISOR, env_status
from .tools.base import specs_for

@asynccontextmanager
async def lifespan(_: FastAPI):
    SUPERVISOR.start_daemon()
    yield
    for env, profile in SUPERVISOR.running():
        SUPERVISOR.stop(env, profile)
    close_shell_sessions()


app = FastAPI(title="Libertarian Agents", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

UI_DIST = Path(__file__).parent.parent / "ui" / "dist"


def _safe_ui_candidate(full_path: str) -> Optional[Path]:
    root = UI_DIST.resolve()
    candidate = (root / full_path).resolve()
    if full_path and root in candidate.parents and candidate.is_file():
        return candidate
    return None


def _board(env: str) -> Board:
    if not control.env_exists(env):
        raise HTTPException(404, f"no such environment: {env}")
    return Board(paths.board_db(env))


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


# ---------------------------------------------------------------- environments


class CreateEnv(BaseModel):
    name: str
    config: EnvConfig = EnvConfig()
    env_file: Optional[str] = None


@app.get("/api/envs")
def list_envs() -> list[dict]:
    out = []
    for e in control.list_envs():
        out.append({**e, "status": env_status(e["name"])})
    return out


@app.post("/api/envs")
def create_env(body: CreateEnv) -> dict:
    try:
        cfg = environment.create_environment(body.name, body.config, env_file=body.env_file)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": body.name, "config": cfg.model_dump()}


@app.get("/api/envs/{env}")
def get_env(env: str) -> dict:
    try:
        cfg = control.get_env(env)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    sb = make_sandbox(env, cfg)
    try:
        sandbox_running = sb.running()
    except RuntimeError:
        sandbox_running = False
    return {
        "name": env,
        "config": cfg.model_dump(),
        "status": env_status(env),
        "env_root": sb.env_root,
        "sandbox_running": sandbox_running,
        "env_file": environment.read_env_file(env),
        "usage": control.usage_for(env).model_dump(),
    }


@app.patch("/api/envs/{env}")
def patch_env(env: str, config: dict = Body(...)) -> dict:
    current_cfg = control.get_env(env)
    payload = dict(config)
    env_file = payload.pop("env_file", None)
    current_env_file = environment.read_env_file(env)
    env_file_changed = env_file is not None and env_file != current_env_file
    current = current_cfg.model_dump()
    cfg = EnvConfig.model_validate({**current, **payload})
    restart_sensitive = {"sandbox", "image", "secrets"}
    restart_changed = any(getattr(cfg, k) != getattr(current_cfg, k) for k in restart_sensitive)
    if restart_changed or env_file_changed:
        if any(e == env for e, _ in SUPERVISOR.running()):
            raise HTTPException(409, "stop the environment before changing sandbox settings")
        close_shell_sessions(env)
    environment.update_environment(env, cfg)
    if env_file is not None:
        environment.write_env_file(env, env_file)
    return {**cfg.model_dump(), "env_file": environment.read_env_file(env)}


@app.delete("/api/envs/{env}")
def delete_env(env: str, keep_files: bool = False) -> dict:
    remaining = SUPERVISOR.stop_env(env, join=15)
    if remaining:
        raise HTTPException(409, f"still stopping: {', '.join(remaining)}; retry deletion shortly")
    try:
        environment.delete_environment(env, remove_files=not keep_files)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": env}


@app.post("/api/envs/{env}/actions/{action}")
def environment_action(env: str, action: str) -> dict:
    try:
        cfg = control.get_env(env)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    sandbox = make_sandbox(env, cfg)
    if action == "start":
        try:
            sandbox.start()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        started = []
        blocked = {}
        for runner in control.list_runners(env):
            if runner.state == "finished":
                continue
            try:
                if runner.state == "waiting":
                    SUPERVISOR.wake(env, runner.profile, "environment started")
                elif not SUPERVISOR.is_running(env, runner.profile):
                    SUPERVISOR.start(env, runner.profile, "environment started")
                started.append(runner.profile)
            except RuntimeError as exc:
                blocked[runner.profile] = str(exc)
        return {
            "ok": True,
            "sandbox_running": True,
            "started": started,
            "blocked": blocked,
        }
    if action == "stop":
        remaining = SUPERVISOR.stop_env(env, join=15)
        if remaining:
            raise HTTPException(409, f"still stopping: {', '.join(remaining)}")
        try:
            sandbox.stop()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"ok": True, "sandbox_running": False, "stopped": True}
    if action == "reset-complete":
        remaining = SUPERVISOR.stop_env(env, join=15)
        if remaining:
            raise HTTPException(409, f"still stopping: {', '.join(remaining)}")
        try:
            environment.reset_environment(env)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "reset": env, "sandbox_running": False}
    raise HTTPException(404, f"unknown environment action: {action}")


# --------------------------------------------------------------------- agents


class CreateAgent(BaseModel):
    name: str
    config: RunnerConfig = RunnerConfig()
    initial_memory: Optional[str] = None


@app.get("/api/envs/{env}/agents")
def list_agents(env: str) -> list[dict]:
    board = _board(env)
    statuses = {a["name"]: a for a in board.list_agents()}
    out = []
    for r in control.list_runners(env):
        used = control.usage_for(env, r.profile)
        out.append(
            {
                "profile": r.profile,
                "state": r.state,
                "running": SUPERVISOR.is_running(env, r.profile),
                "wake_at": r.wake_at,
                "updated_at": r.updated_at,
                "stop_reason": r.stop_reason,
                "status": (statuses.get(r.profile) or {}).get("status", ""),
                "config": r.config.model_dump(),
                "usage": used.model_dump(),
                "unread": board.unread_count(r.profile),
            }
        )
    return out


@app.post("/api/envs/{env}/agents")
def create_agent(env: str, body: CreateAgent) -> dict:
    try:
        cfg = environment.create_profile(
            env, body.name, body.config, initial_memory=body.initial_memory
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"profile": body.name, "config": cfg.model_dump()}


@app.get("/api/envs/{env}/agents/{profile}")
def get_agent(env: str, profile: str) -> dict:
    try:
        r = control.get_runner(env, profile)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    conv_path = paths.history_dir(env, profile) / "conversation.json"
    conv = json.loads(_read(conv_path) or "{}")
    cfg = control.get_env(env)
    sandbox = make_sandbox(env, cfg)
    memory = _read(paths.memory_file(env, profile))
    system_prompt = prompts.instructions(
        profile,
        sandbox.env_root,
        r.config.memory_char_limit,
        r.config.base_prompt_override,
    )
    tool_definitions = []
    if r.config.provider == "openai":
        from .providers.openai_provider import tool_payload
    else:
        from .providers.openrouter_provider import tool_payload
    defaults = {spec.name: spec.description for spec in specs_for(r.config.tools)}
    for spec in specs_for(r.config.tools, r.config.tool_description_overrides):
        hosted = spec.name == "web_search"
        tool_definitions.append(
            {
                "name": spec.name,
                "logical_description": spec.description,
                "default_description": defaults[spec.name],
                "transport": "hosted" if hosted else "application function",
                "provider_payload": tool_payload(spec),
            }
        )
    return {
        "profile": profile,
        "state": r.state,
        "running": SUPERVISOR.is_running(env, profile),
        "wake_at": r.wake_at,
        "updated_at": r.updated_at,
        "stop_reason": r.stop_reason,
        "config": r.config.model_dump(),
        "usage": control.usage_for(env, profile).model_dump(),
        "memory": memory,
        "prompt": {
            "system_prompt": system_prompt,
            "using_base_override": bool(
                r.config.base_prompt_override and r.config.base_prompt_override.strip()
            ),
            "tools": tool_definitions,
            "conversation_items": conv.get("items", []),
            "injected": [
                {
                    "name": "Agent goal",
                    "timing": "conversation prefix; refreshed after compaction",
                    "content": prompts.GOAL_BLOCK.format(goal=r.config.goal or "(none)"),
                },
                {
                    "name": "Agent goal update",
                    "timing": "appended on the next wake after an operator changes the goal",
                    "content": prompts.GOAL_UPDATE_BLOCK,
                },
                {
                    "name": "memory.md snapshot",
                    "timing": "conversation prefix; frozen between compactions",
                    "content": prompts.MEMORY_BLOCK.format(memory=memory or "(empty)"),
                },
                {"name": "Wake message", "timing": "each wake", "content": prompts.WAKE},
                {
                    "name": "Tool-result status prefix",
                    "timing": "prepended to every application tool result",
                    "content": prompts.STATUS,
                },
                {
                    "name": "Post-compaction continuation",
                    "timing": "after manual context compaction",
                    "content": prompts.CONTEXT_COMPACTED,
                },
                {
                    "name": "No-local-tool nudge",
                    "timing": "when a turn has no application tool call",
                    "content": prompts.NO_TOOL_CALL,
                },
                {
                    "name": "Budget warning",
                    "timing": "when the input/environment budget is exhausted",
                    "content": prompts.BUDGET_EXHAUSTED,
                },
                {
                    "name": "Manual compaction request",
                    "timing": "manual compaction only",
                    "content": prompts.COMPACTION_PROMPT,
                },
            ],
        },
        "context": {
            "items": len(conv.get("items", [])),
            "compactions": conv.get("compactions", 0),
            "last_input_tokens": conv.get("last_input_tokens", 0),
            "summary": conv.get("summary"),
        },
    }


@app.patch("/api/envs/{env}/agents/{profile}")
def patch_agent(env: str, profile: str, body: dict = Body(...)) -> dict:
    r = control.get_runner(env, profile)
    warnings = []
    if "config" in body:
        if r.state == "active" or SUPERVISOR.is_running(env, profile):
            raise HTTPException(409, "stop the runner before changing its config")
        cfg = RunnerConfig.model_validate({**r.config.model_dump(), **body["config"]})
        if cfg.model != r.config.model or cfg.provider != r.config.provider:
            warnings.append(
                "Model changed: the next run compacts first and reasoning context is lost."
            )
        control.upsert_runner(env, profile, cfg)
    if "memory" in body:
        limit = control.get_runner(env, profile).config.memory_char_limit
        if len(body["memory"]) > limit:
            raise HTTPException(400, f"memory.md exceeds its {limit}-character limit")
        paths.memory_file(env, profile).write_text(body["memory"], encoding="utf-8")
    return {"ok": True, "warnings": warnings}


@app.delete("/api/envs/{env}/agents/{profile}")
def delete_agent(env: str, profile: str, keep_files: bool = False) -> dict:
    try:
        stopped = SUPERVISOR.stop(env, profile, join=15)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not stopped:
        raise HTTPException(409, "runner is still stopping; retry deletion shortly")
    try:
        environment.delete_profile(env, profile, remove_files=not keep_files)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": profile}


@app.post("/api/envs/{env}/agents/{profile}/{action}")
def agent_action(env: str, profile: str, action: str, reason: str = "operator") -> dict:
    try:
        control.get_runner(env, profile)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if action == "start":
        try:
            SUPERVISOR.start(env, profile, f"started by {reason}")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    elif action == "stop":
        SUPERVISOR.stop(env, profile)
    elif action == "wake":
        try:
            SUPERVISOR.wake(env, profile, f"woken by {reason}")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    elif action == "reset":
        if SUPERVISOR.is_running(env, profile):
            raise HTTPException(409, "stop the runner first")
        (paths.history_dir(env, profile) / "conversation.json").unlink(missing_ok=True)
        control.set_state(env, profile, "inactive", stop_reason="context reset")
    elif action == "reset-complete":
        if not SUPERVISOR.stop(env, profile, join=15):
            raise HTTPException(409, "runner is still stopping; retry reset shortly")
        try:
            environment.reset_profile(env, profile)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        raise HTTPException(404, f"unknown action: {action}")
    return {"ok": True, "state": control.get_runner(env, profile).state}


@app.get("/api/envs/{env}/agents/{profile}/events")
def agent_events(env: str, profile: str, after: int = -1, limit: int = 300) -> list[dict]:
    log = EventLog(paths.history_dir(env, profile) / "events.jsonl")
    return log.tail(limit=limit, after=after if after >= 0 else None)


# ---------------------------------------------------------------------- board


class SendBody(BaseModel):
    to: str
    body: str
    sender: str = USER


@app.get("/api/envs/{env}/board")
def board_messages(env: str, after: int = 0, limit: int = 300) -> list[dict]:
    return [m.__dict__ for m in _board(env).recent(limit=limit, after=after)]


@app.get("/api/envs/{env}/channels")
def channels(env: str) -> dict:
    board = _board(env)
    return {"channels": board.list_channels(), "agents": board.list_agents()}


@app.post("/api/envs/{env}/board/send")
def send(env: str, body: SendBody) -> dict:
    cfg = control.get_env(env)
    board = _board(env)
    mid, truncated = board.send(
        body.sender, body.to, body.body,
        max_chars=cfg.max_message_chars, spill_dir=paths.shared_dir(env) / "messages",
    )
    woken = []
    channel = body.to[1:] if body.to.startswith("#") else None
    recipient = None if channel is not None else body.to.lstrip("@")
    subscribers = board.subscribers(channel) if channel is not None else set()
    for r in control.list_runners(env):
        addressed = r.profile in subscribers if channel is not None else recipient == r.profile
        if r.state == "waiting" and addressed:
            SUPERVISOR.wake(env, r.profile, "new message")
            woken.append(r.profile)
    return {"id": mid, "truncated": truncated, "woken": woken}


# ---------------------------------------------------------- files and console


@app.get("/api/envs/{env}/files")
def files(env: str, path: str = "") -> dict:
    root = paths.env_dir(env)
    mapper = environment.path_mapper(env)
    try:
        target = mapper.to_host(path or mapper.env_root)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not target.exists():
        raise HTTPException(404, "not found")
    if target.is_dir():
        entries = [
            {
                "name": p.name,
                "dir": p.is_dir(),
                "size": p.stat().st_size if p.is_file() else None,
                "path": str(p.relative_to(root)),
            }
            for p in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        ]
        return {"path": str(target.relative_to(root)) if target != root else "", "dir": True, "entries": entries}
    data = target.read_bytes()[:200_000]
    return {
        "path": str(target.relative_to(root)),
        "dir": False,
        "content": data.decode("utf-8", "replace"),
        "size": target.stat().st_size,
    }


@app.post("/api/envs/{env}/exec")
def exec_command(env: str, body: dict = Body(...)) -> dict:
    """Operator console into the sandbox. Uncompressed, unlike the agent tool."""
    try:
        cfg = control.get_env(env)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    sb = make_sandbox(env, cfg)
    try:
        sb.start()
        result = sb.exec(
            body.get("command", ""), cwd=body.get("cwd") or sb.env_root, timeout=60
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"exit_code": result.exit_code, "output": result.output[-50_000:], "timed_out": result.timed_out}


@app.get("/api/envs/{env}/usage")
def usage(env: str) -> dict:
    return {"total": control.usage_for(env).model_dump(), "breakdown": control.usage_breakdown(env)}


# ------------------------------------------------------------------- streaming


@app.get("/api/envs/{env}/stream")
async def stream(env: str, after: int = Query(0)) -> StreamingResponse:
    """Poll-backed SSE: new board messages and agent state changes."""

    async def gen():
        last_msg = after
        last_state: dict[str, Any] = {}
        while True:
            try:
                board = Board(paths.board_db(env))
                msgs = board.recent(limit=100, after=last_msg)
                if msgs:
                    last_msg = msgs[-1].id
                    payload = json.dumps({"type": "messages", "messages": [m.__dict__ for m in msgs]})
                    yield f"data: {payload}\n\n"
                agents = list_agents(env)
                snapshot = {a["profile"]: (a["state"], a["status"], a["usage"]["input_tokens"]) for a in agents}
                if snapshot != last_state:
                    last_state = snapshot
                    yield f"data: {json.dumps({'type': 'agents', 'agents': agents, 'status': env_status(env)})}\n\n"
            except Exception as exc:  # never kill the stream on a transient error
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/tools")
def tools() -> list[dict]:
    from .tools.base import REGISTRY, load_all

    load_all()
    return [{"name": t.name, "description": t.description} for t in REGISTRY.values()]


# ------------------------------------------------------------------- static UI

if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = _safe_ui_candidate(full_path)
        if candidate is not None:
            return FileResponse(candidate)
        return FileResponse(UI_DIST / "index.html")
