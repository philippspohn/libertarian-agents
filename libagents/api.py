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

from . import control, environment, paths
from .board import USER, Board
from .events import EventLog
from .models import EnvConfig, RunnerConfig
from .sandbox import make_sandbox
from .supervisor import SUPERVISOR, env_status

@asynccontextmanager
async def lifespan(_: FastAPI):
    SUPERVISOR.start_daemon()
    yield
    for env, profile in SUPERVISOR.running():
        SUPERVISOR.stop(env, profile)


app = FastAPI(title="Libertarian Agents", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

UI_DIST = Path(__file__).parent.parent / "ui" / "dist"


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


@app.get("/api/envs")
def list_envs() -> list[dict]:
    out = []
    for e in control.list_envs():
        out.append({**e, "status": env_status(e["name"])})
    return out


@app.post("/api/envs")
def create_env(body: CreateEnv) -> dict:
    try:
        cfg = environment.create_environment(body.name, body.config)
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
    return {
        "name": env,
        "config": cfg.model_dump(),
        "status": env_status(env),
        "env_root": sb.env_root,
        "sandbox_running": sb.running(),
        "usage": control.usage_for(env).model_dump(),
    }


@app.patch("/api/envs/{env}")
def patch_env(env: str, config: dict = Body(...)) -> dict:
    current = control.get_env(env).model_dump()
    cfg = EnvConfig.model_validate({**current, **config})
    environment.update_environment(env, cfg)
    return cfg.model_dump()


@app.delete("/api/envs/{env}")
def delete_env(env: str, keep_files: bool = False) -> dict:
    SUPERVISOR.stop_env(env)
    environment.delete_environment(env, remove_files=not keep_files)
    return {"deleted": env}


# --------------------------------------------------------------------- agents


class CreateAgent(BaseModel):
    name: str
    config: RunnerConfig = RunnerConfig()
    agent_md: Optional[str] = None


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
        cfg = environment.create_profile(env, body.name, body.config, agent_md=body.agent_md)
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
    return {
        "profile": profile,
        "state": r.state,
        "running": SUPERVISOR.is_running(env, profile),
        "wake_at": r.wake_at,
        "stop_reason": r.stop_reason,
        "config": r.config.model_dump(),
        "usage": control.usage_for(env, profile).model_dump(),
        "agent_md": _read(paths.agent_md(env, profile)),
        "memory": _read(paths.memory_file(env, profile)),
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
    if "agent_md" in body:
        paths.agent_md(env, profile).write_text(body["agent_md"], encoding="utf-8")
    if "memory" in body:
        paths.memory_file(env, profile).write_text(body["memory"], encoding="utf-8")
    return {"ok": True, "warnings": warnings}


@app.delete("/api/envs/{env}/agents/{profile}")
def delete_agent(env: str, profile: str, keep_files: bool = False) -> dict:
    SUPERVISOR.stop(env, profile)
    environment.delete_profile(env, profile, remove_files=not keep_files)
    return {"deleted": profile}


@app.post("/api/envs/{env}/agents/{profile}/{action}")
def agent_action(env: str, profile: str, action: str, reason: str = "operator") -> dict:
    if action == "start":
        try:
            SUPERVISOR.start(env, profile, f"started by {reason}")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    elif action == "stop":
        SUPERVISOR.stop(env, profile)
    elif action == "wake":
        SUPERVISOR.wake(env, profile, f"woken by {reason}")
    elif action == "reset":
        if SUPERVISOR.is_running(env, profile):
            raise HTTPException(409, "stop the runner first")
        (paths.history_dir(env, profile) / "conversation.json").unlink(missing_ok=True)
        control.set_state(env, profile, "inactive", stop_reason="context reset")
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
    mid, truncated = _board(env).send(
        body.sender, body.to, body.body,
        max_chars=cfg.max_message_chars, spill_dir=paths.shared_dir(env) / "messages",
    )
    woken = []
    channel = body.to.startswith("#")
    recipient = None if channel else body.to.lstrip("@")
    for r in control.list_runners(env):
        if r.state == "waiting" and (channel or recipient == r.profile):
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
    cfg = control.get_env(env)
    sb = make_sandbox(env, cfg)
    sb.start()
    result = sb.exec(body.get("command", ""), cwd=body.get("cwd") or sb.env_root, timeout=60)
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
        candidate = UI_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(UI_DIST / "index.html")
