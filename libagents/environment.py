"""Creating and tearing down environments and agent profiles.

Creating a profile is `mkdir` plus two files. There is no profile record in
the control DB beyond its runner config, so the directory listing is the
source of truth for who exists.
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

DEFAULT_AGENT_MD = """\
# {profile}

_Your own standing instructions. Rewrite this file however you like -- it is
loaded into your prompt at the start of every run._

## Who I am

(unwritten)

## How I work

(unwritten)
"""

DEFAULT_MEMORY = """\
# memory.md

_Your state snapshot. Survives compaction; nothing else does._

## Current focus

(nothing yet)

## Facts worth keeping

(nothing yet)

## Open threads

(nothing yet)
"""


def validate_name(name: str, kind: str = "name") -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"invalid {kind} {name!r}: use lowercase letters, digits, '-' or '_' (max 32)")
    return name


def create_environment(name: str, config: EnvConfig | None = None) -> EnvConfig:
    validate_name(name, "environment name")
    if control.env_exists(name):
        raise ValueError(f"environment {name!r} already exists")
    config = config or EnvConfig()
    paths.agents_dir(name).mkdir(parents=True, exist_ok=True)
    paths.shared_dir(name).mkdir(parents=True, exist_ok=True)
    control.create_env(name, config)
    Board(paths.board_db(name)).init()
    write_goal(name, config.goal)
    return config


def write_goal(env: str, goal: str) -> None:
    (paths.shared_dir(env) / "GOAL.md").write_text(goal or "(no goal set)\n", encoding="utf-8")


def update_environment(name: str, config: EnvConfig) -> None:
    control.set_env(name, config)
    write_goal(name, config.goal)


def delete_environment(name: str, *, remove_files: bool = True) -> None:
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
    agent_md: str | None = None,
) -> RunnerConfig:
    validate_name(profile, "profile name")
    if not control.env_exists(env):
        raise KeyError(f"no such environment: {env}")
    d = paths.profile_dir(env, profile)
    d.mkdir(parents=True, exist_ok=True)
    (d / "history").mkdir(exist_ok=True)
    (d / "outputs").mkdir(exist_ok=True)
    am = paths.agent_md(env, profile)
    if not am.exists():
        am.write_text(agent_md or DEFAULT_AGENT_MD.format(profile=profile), encoding="utf-8")
    mem = paths.memory_file(env, profile)
    if not mem.exists():
        mem.write_text(DEFAULT_MEMORY, encoding="utf-8")

    config = config or RunnerConfig()
    control.upsert_runner(env, profile, config)
    Board(paths.board_db(env)).register(profile)
    return config


def delete_profile(env: str, profile: str, *, remove_files: bool = True) -> None:
    control.delete_runner(env, profile)
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
