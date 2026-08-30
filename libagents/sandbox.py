"""Sandboxes.

One container per ENVIRONMENT, not per agent -- a shared read/write
filesystem is the whole premise, so isolating agents from each other would
defeat it. The environment directory is bind-mounted at `/env`, and file
tools operate directly on the host path (fast, and the UI can read the same
bytes). Only shell commands go through the sandbox boundary.

`LocalSandbox` is a dev fallback that runs commands as plain subprocesses in
the environment directory. It is NOT isolation -- use it only on a machine
you would let the agents touch anyway.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import paths
from .models import EnvConfig


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


class Sandbox(Protocol):
    env_root: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def running(self) -> bool: ...
    def exec(self, command: str, cwd: str, timeout: int) -> ExecResult: ...


def _secret_env(config: EnvConfig) -> dict[str, str]:
    return {k: os.environ[k] for k in config.secrets if k in os.environ}


class LocalSandbox:
    """Runs commands on the host, rooted at the environment directory."""

    def __init__(self, env: str, config: EnvConfig):
        self.env = env
        self.config = config
        self.host_root = paths.env_dir(env)
        self.env_root = str(self.host_root)

    def start(self) -> None:
        self.host_root.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        pass

    def running(self) -> bool:
        return self.host_root.exists()

    def exec(self, command: str, cwd: str, timeout: int) -> ExecResult:
        workdir = Path(cwd) if Path(cwd).is_absolute() else self.host_root / cwd
        if not workdir.exists():
            workdir = self.host_root
        environ = {**os.environ, **_secret_env(self.config), "ENV_ROOT": self.env_root}
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=workdir,
                env=environ,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            return ExecResult(124, partial, timed_out=True)
        return ExecResult(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


class DockerSandbox:
    """One long-lived container per environment."""

    def __init__(self, env: str, config: EnvConfig):
        self.env = env
        self.config = config
        self.host_root = paths.env_dir(env)
        self.env_root = paths.DOCKER_ROOT
        self.container = f"libagents-{env}"

    def _docker(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        if not shutil.which("docker"):
            raise RuntimeError("docker not found on PATH; set the environment's sandbox to 'local'")
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)

    def running(self) -> bool:
        r = self._docker("inspect", "-f", "{{.State.Running}}", self.container)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def start(self) -> None:
        self.host_root.mkdir(parents=True, exist_ok=True)
        if self.running():
            return
        exists = self._docker("inspect", self.container).returncode == 0
        if exists:
            self._docker("start", self.container)
            return
        args = [
            "run", "-d", "--name", self.container,
            "-v", f"{self.host_root}:{paths.DOCKER_ROOT}",
            "-w", paths.DOCKER_ROOT,
            "-e", f"ENV_ROOT={paths.DOCKER_ROOT}",
        ]
        for k, v in _secret_env(self.config).items():
            args += ["-e", f"{k}={v}"]
        args += [self.config.image, "sleep", "infinity"]
        r = self._docker(*args, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"failed to start container: {r.stderr.strip()}")

    def stop(self) -> None:
        self._docker("stop", "-t", "5", self.container, timeout=60)

    def destroy(self) -> None:
        self.stop()
        self._docker("rm", "-f", self.container)

    def exec(self, command: str, cwd: str, timeout: int) -> ExecResult:
        if not self.running():
            self.start()
        try:
            r = self._docker(
                "exec", "-w", cwd, self.container, "/bin/sh", "-lc", command,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return ExecResult(124, out, timed_out=True)
        return ExecResult(r.returncode, (r.stdout or "") + (r.stderr or ""))


def make_sandbox(env: str, config: EnvConfig) -> Sandbox:
    if config.sandbox == "docker":
        return DockerSandbox(env, config)
    return LocalSandbox(env, config)
