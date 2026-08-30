"""Filesystem layout of an environment.

Host side:                          Agent-facing (`env_root`):
  <home>/envs/<env>/agents/alice/     <root>/agents/alice/
  <home>/envs/<env>/shared/           <root>/shared/

`env_root` is `/env` under Docker (the whole environment dir is bind-mounted
there) and the host path itself under the local sandbox. Every tool accepts
either an absolute path under the root or a path relative to it.
"""

from __future__ import annotations

import os
from pathlib import Path

DOCKER_ROOT = "/env"


def home() -> Path:
    return Path(os.environ.get("LIBAGENTS_HOME", Path.home() / ".libertarian-agents"))


def control_db() -> Path:
    return home() / "control.db"


def env_dir(env: str) -> Path:
    return home() / "envs" / env


def agents_dir(env: str) -> Path:
    return env_dir(env) / "agents"


def profile_dir(env: str, profile: str) -> Path:
    return agents_dir(env) / profile


def shared_dir(env: str) -> Path:
    return env_dir(env) / "shared"


def board_db(env: str) -> Path:
    return shared_dir(env) / "board.db"


def spill_dir(env: str, profile: str) -> Path:
    return profile_dir(env, profile) / "outputs"


def history_dir(env: str, profile: str) -> Path:
    return profile_dir(env, profile) / "history"


def memory_file(env: str, profile: str) -> Path:
    return profile_dir(env, profile) / "memory.md"


def agent_md(env: str, profile: str) -> Path:
    return profile_dir(env, profile) / "AGENT.md"


class PathMapper:
    """Translates between agent-facing paths and host paths, and refuses to
    resolve anything outside the environment root."""

    def __init__(self, host_root: Path, env_root: str):
        self.host_root = host_root.resolve()
        self.env_root = env_root.rstrip("/") or "/"

    def to_host(self, path: str) -> Path:
        p = (path or "").strip()
        if not p:
            raise ValueError("empty path")
        if self.env_root != "/" and (p == self.env_root or p.startswith(self.env_root + "/")):
            p = p[len(self.env_root):]
        elif p.startswith(str(self.host_root)):
            p = p[len(str(self.host_root)):]
        p = p.lstrip("/")
        resolved = (self.host_root / p).resolve() if p else self.host_root
        if resolved != self.host_root and self.host_root not in resolved.parents:
            raise ValueError(f"path escapes the environment root: {path}")
        return resolved

    def to_env(self, host_path: Path) -> str:
        rel = host_path.resolve().relative_to(self.host_root)
        base = "" if self.env_root == "/" else self.env_root
        return f"{base}/{rel}" if str(rel) != "." else base or "/"
