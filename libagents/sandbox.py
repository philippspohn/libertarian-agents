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
import selectors
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import dotenv_values

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
    def exec(
        self, command: str, cwd: str, timeout: int, session: str | None = None
    ) -> ExecResult: ...


class _ShellSession:
    """A long-lived, non-interactive shell with delimiter-framed commands."""

    def __init__(self, argv: list[str], cwd: str | Path, env: dict[str, str] | None = None):
        self.proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if self.proc.stdout is None or self.proc.stdin is None:
            raise RuntimeError("failed to open persistent shell pipes")
        os.set_blocking(self.proc.stdout.fileno(), False)
        self.pending = b""
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _returncode(self) -> int:
        return self.proc.returncode if self.proc.returncode is not None else 1

    def run(self, command: str, timeout: int) -> ExecResult:
        with self.lock:
            if not self.alive():
                return ExecResult(self._returncode(), self.pending.decode("utf-8", "replace"))
            token = f"__LIBAGENTS_DONE_{uuid.uuid4().hex}__".encode()
            marker = b"\n" + token + b" "
            payload = (
                command.encode("utf-8")
                + b"\n__libagents_rc=$?\nprintf '\\n"
                + token
                + b" %s\\n' \"$__libagents_rc\"\n"
            )
            try:
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.proc.poll()
                return ExecResult(self._returncode(), self.pending.decode("utf-8", "replace"))

            data = self.pending
            self.pending = b""
            selector = selectors.DefaultSelector()
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            try:
                while True:
                    start = data.find(marker)
                    if start >= 0:
                        end = data.find(b"\n", start + len(marker))
                        if end >= 0:
                            raw_rc = data[start + len(marker):end].strip()
                            try:
                                rc = int(raw_rc)
                            except ValueError:
                                rc = 1
                            output = data[:start]
                            # The protocol inserts one newline before its marker.
                            if output.endswith(b"\n"):
                                output = output[:-1]
                            self.pending = data[end + 1:]
                            return ExecResult(rc, output.decode("utf-8", "replace"))

                    if not self.alive():
                        try:
                            data += os.read(self.proc.stdout.fileno(), 65536)
                        except (BlockingIOError, OSError):
                            pass
                        return ExecResult(
                            self._returncode(), data.decode("utf-8", "replace")
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.close()
                        return ExecResult(124, data.decode("utf-8", "replace"), timed_out=True)
                    for _, _ in selector.select(min(0.1, remaining)):
                        try:
                            chunk = os.read(self.proc.stdout.fileno(), 65536)
                        except BlockingIOError:
                            chunk = b""
                        if chunk:
                            data += chunk
            finally:
                selector.close()

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=1)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


_sessions: dict[tuple[str, str, str], _ShellSession] = {}
_sessions_lock = threading.Lock()


def close_shell_sessions(env: str | None = None, profile: str | None = None) -> None:
    """Close persistent shells matching an environment/profile."""
    with _sessions_lock:
        keys = [
            key for key in _sessions
            if (env is None or key[1] == env) and (profile is None or key[2] == profile)
        ]
        sessions = [_sessions.pop(key) for key in keys]
    for session in sessions:
        session.close()


def _persistent_session(
    key: tuple[str, str, str], factory
) -> _ShellSession:
    with _sessions_lock:
        current = _sessions.get(key)
        if current is None or not current.alive():
            if current is not None:
                current.close()
            current = factory()
            _sessions[key] = current
        return current


def _sandbox_env(env: str, config: EnvConfig) -> dict[str, str]:
    """Environment-local .env values plus explicitly forwarded host values.

    Host-forwarded names win when both sources define the same key.
    """
    values = dotenv_values(paths.env_file(env), interpolate=False)
    local = {k: v for k, v in values.items() if v is not None}
    forwarded = {k: os.environ[k] for k in config.secrets if k in os.environ}
    return {**local, **forwarded}


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
        close_shell_sessions(self.env)

    def running(self) -> bool:
        return self.host_root.exists()

    def exec(
        self, command: str, cwd: str, timeout: int, session: str | None = None
    ) -> ExecResult:
        workdir = Path(cwd) if Path(cwd).is_absolute() else self.host_root / cwd
        if not workdir.exists():
            workdir = self.host_root
        environ = {**os.environ, **_sandbox_env(self.env, self.config), "ENV_ROOT": self.env_root}
        if session:
            shell = _persistent_session(
                ("local", self.env, session),
                lambda: _ShellSession(
                    ["/bin/bash", "--noprofile", "--norc"], workdir, environ
                ),
            )
            return shell.run(command, timeout)
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
        for k, v in _sandbox_env(self.env, self.config).items():
            args += ["-e", f"{k}={v}"]
        args += [self.config.image, "sleep", "infinity"]
        r = self._docker(*args, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"failed to start container: {r.stderr.strip()}")

    def stop(self) -> None:
        close_shell_sessions(self.env)
        self._docker("stop", "-t", "5", self.container, timeout=60)

    def destroy(self) -> None:
        if not shutil.which("docker"):
            return
        if self.running():
            self.stop()
        if self._docker("inspect", self.container).returncode == 0:
            self._docker("rm", "-f", self.container)

    def exec(
        self, command: str, cwd: str, timeout: int, session: str | None = None
    ) -> ExecResult:
        if not self.running():
            self.start()
        if session:
            def create_session() -> _ShellSession:
                args = ["docker", "exec", "-i", "-w", cwd]
                # Use the current .env/forwarded values, rather than the values
                # captured when the long-lived container was first created.
                for key, value in _sandbox_env(self.env, self.config).items():
                    args += ["-e", f"{key}={value}"]
                args += [self.container, "/bin/sh"]
                return _ShellSession(args, self.host_root)

            shell = _persistent_session(("docker", self.env, session), create_session)
            return shell.run(command, timeout)
        try:
            args = ["exec", "-w", cwd]
            for key, value in _sandbox_env(self.env, self.config).items():
                args += ["-e", f"{key}={value}"]
            args += [self.container, "/bin/sh", "-lc", command]
            r = self._docker(*args, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return ExecResult(124, out, timed_out=True)
        return ExecResult(r.returncode, (r.stdout or "") + (r.stderr or ""))


def make_sandbox(env: str, config: EnvConfig) -> Sandbox:
    if config.sandbox == "docker":
        return DockerSandbox(env, config)
    return LocalSandbox(env, config)
