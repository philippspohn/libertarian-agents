"""Creating and tearing down environments and agent profiles.

Creating a profile writes its host-controlled runner config to control.db and
creates its agent-visible files inside the environment.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from . import control, paths
from .board import Board
from .models import EnvConfig, RunnerConfig
from .sandbox import DockerSandbox, make_sandbox

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

DEFAULT_MEMORY = """\
# memory.md

*Your state snapshot. Survives compaction; nothing else does.*
"""

DEFAULT_ENV_FILE = """\
# Environment-scoped variables available to every agent shell.
# This file is plain text inside the shared environment; do not put anything
# here that agents in this trusted environment should not be able to read.
"""


def validate_name(name: str, kind: str = "name") -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"invalid {kind} {name!r}: use lowercase letters, digits, '-' or '_' (max 32)")
    return name


def create_environment(
    name: str, config: EnvConfig | None = None, *, env_file: str | None = None
) -> EnvConfig:
    validate_name(name, "environment name")
    if control.env_exists(name):
        raise ValueError(f"environment {name!r} already exists")
    config = config or EnvConfig()
    paths.agents_dir(name).mkdir(parents=True, exist_ok=True)
    paths.shared_dir(name).mkdir(parents=True, exist_ok=True)
    control.create_env(name, config)
    Board(paths.board_db(name)).init()
    write_env_file(name, DEFAULT_ENV_FILE if env_file is None else env_file)
    return config


def read_env_file(env: str) -> str:
    fp = paths.env_file(env)
    return fp.read_text(encoding="utf-8", errors="replace") if fp.exists() else ""


def write_env_file(env: str, content: str) -> None:
    paths.env_file(env).write_text(content, encoding="utf-8")


def update_environment(name: str, config: EnvConfig) -> None:
    validate_name(name, "environment name")
    old = control.get_env(name)
    if old.sandbox == "docker" and (
        config.sandbox != old.sandbox
        or config.image != old.image
    ):
        DockerSandbox(name, old).destroy()
    control.set_env(name, config)


def delete_environment(name: str, *, remove_files: bool = True) -> None:
    validate_name(name, "environment name")
    from .sandbox import close_shell_sessions
    close_shell_sessions(name)
    try:
        cfg = control.get_env(name)
        if cfg.sandbox == "docker":
            DockerSandbox(name, cfg).destroy()
    except Exception:
        pass
    control.delete_env(name)
    if remove_files and paths.env_dir(name).exists():
        shutil.rmtree(paths.env_dir(name))


def list_profiles(env: str) -> list[str]:
    root = paths.agents_dir(env)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def create_profile(
    env: str,
    profile: str,
    config: RunnerConfig | None = None,
    initial_memory: str | None = None,
) -> RunnerConfig:
    validate_name(profile, "profile name")
    if not control.env_exists(env):
        raise KeyError(f"no such environment: {env}")
    if control.runner_exists(env, profile):
        raise ValueError(f"profile {profile!r} already exists in {env!r}")
    d = paths.profile_dir(env, profile)
    d.mkdir(parents=True, exist_ok=True)
    (d / "history").mkdir(exist_ok=True)
    (d / "outputs").mkdir(exist_ok=True)
    toolkit = d / "toolkit"
    toolkit.mkdir(exist_ok=True)
    toolkit_readme = toolkit / "README.md"
    if not toolkit_readme.exists():
        toolkit_readme.write_text(
            "# Toolkit\n\nProfile-local scripts, notes, and reusable helpers belong here.\n",
            encoding="utf-8",
        )
    mem = paths.memory_file(env, profile)
    if not mem.exists():
        content = DEFAULT_MEMORY if initial_memory is None else initial_memory
        limit = (config or RunnerConfig()).memory_char_limit
        if len(content) > limit:
            raise ValueError(f"initial memory.md exceeds its {limit}-character limit")
        mem.write_text(content, encoding="utf-8")

    config = config or RunnerConfig()
    control.upsert_runner(env, profile, config)
    Board(paths.board_db(env)).register(profile)
    return config


def delete_profile(env: str, profile: str, *, remove_files: bool = True) -> None:
    validate_name(env, "environment name")
    validate_name(profile, "profile name")
    control.delete_runner(env, profile)
    from .sandbox import close_shell_sessions
    close_shell_sessions(env, profile)
    Board(paths.board_db(env)).unregister(profile)
    d = paths.profile_dir(env, profile)
    if remove_files and d.exists():
        shutil.rmtree(d)


def start_sandbox(env: str):
    cfg = control.get_env(env)
    sb = make_sandbox(env, cfg)
    sb.start()
    return sb


def path_mapper(env: str):
    cfg = control.get_env(env)
    sb = make_sandbox(env, cfg)
    return paths.PathMapper(paths.env_dir(env), sb.env_root)
