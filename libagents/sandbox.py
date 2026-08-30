"""Sandboxes.

One container per ENVIRONMENT, not per agent -- a shared read/write
filesystem is the whole premise, so isolating agents from each other would
defeat it. The environment directory is bind-mounted at `/env`, and file
tools operate directly on the host path (fast, and the UI can read the same
bytes). Only shell commands go through the sandbox boundary.

`MacOSSandbox` keeps host-native execution (including Metal/MLX) while using
Apple Seatbelt to confine filesystem writes to the environment. It is a
best-effort guard against accidental damage, not a hostile-code boundary.

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
import sys
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


# Read access is deliberately broad so host-native compilers, Homebrew tools,
# Python environments, and model weights keep working. Writes are narrowly
# scoped below. system-graphics is Apple's own GPU/Metal policy fragment.
MACOS_SEATBELT_PROFILE = """\
(version 1)
(deny default)
(import "system.sb")
(import "com.apple.corefoundation.sb")
(corefoundation)
(system-graphics)

(allow file-read*)
(allow file-write* (subpath (param "ENV_ROOT")))

; Xcode and SwiftPM ignore TMPDIR for a few atomic caches. Permit only the
; cache names they create inside this user's kernel-provided Darwin folders.
(allow file-write*
  (require-all
    (subpath (param "DARWIN_TEMP"))
    (require-any
      (regex #"/xcrun_db(-[^/]+)?$")
      (regex #"/CFNetworkDownload_[^/]+[.]tmp$")
      (regex #"/TemporaryItems/NSIRD_swift-(build|driver|frontend)_[^/]+(/.*)?$")
    )
  )
)
(allow file-write*
  (require-all
    (subpath (param "DARWIN_CACHE"))
    (regex #"/com[.]apple[.]DeveloperTools/[^/]+/Xcode/PlugInCache-xcodebuild[.]xcplugincache$")
  )
)
; The Metal frontend also ignores TMPDIR and creates compiler modules in its
; fixed per-user cache. Limit the exception to that one cache subtree.
(allow file-write* (subpath (param "DARWIN_METAL_CACHE")))

(allow process-fork process-exec)
(allow process-info* (target self))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm*)
(allow network*)
"""


# These Darwin-only confstr constants are published in Apple's unistd.h but
# are not included in Python's os.confstr_names mapping.
_CS_DARWIN_USER_TEMP_DIR = 65537


def _darwin_user_write_dirs() -> tuple[Path, Path]:
    """Return canonical, kernel-provided per-user Darwin temp/cache folders."""
    raw_temp = os.confstr(_CS_DARWIN_USER_TEMP_DIR)
    if not raw_temp:
        raise RuntimeError("macOS did not provide a Darwin user temporary directory")
    temp = Path(raw_temp).resolve()
    # The cache directory is the T directory's C sibling. macOS currently
    # returns EIO for _CS_DARWIN_USER_CACHE_DIR on some releases, so deriving
    # it from the kernel-provided temp path is more reliable.
    if temp.name != "T" or temp.parent.name in {"", ".", ".."}:
        raise RuntimeError(f"unexpected Darwin user temporary directory: {temp}")
    return temp, temp.with_name("C")


def _host_workdir(host_root: Path, cwd: str) -> Path:
    """Resolve an initial host cwd, falling back if it is absent or outside."""
    root = host_root.resolve()
    candidate = Path(cwd) if Path(cwd).is_absolute() else root / cwd
    try:
        candidate = candidate.resolve()
        if candidate != root and root not in candidate.parents:
            return root
    except OSError:
        return root
    return candidate if candidate.exists() else root


def _active_marker(host_root: Path) -> Path:
    return host_root / ".sandbox" / "active.pid"


def _mark_started(host_root: Path) -> None:
    marker = _active_marker(host_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(os.getpid()), encoding="ascii")


def _mark_stopped(host_root: Path) -> None:
    _active_marker(host_root).unlink(missing_ok=True)


def _marked_running(host_root: Path) -> bool:
    try:
        pid = int(_active_marker(host_root).read_text(encoding="ascii"))
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError, OSError):
        return False


def _macos_host_env(env: str, config: EnvConfig, host_root: Path) -> dict[str, str]:
    """Small host environment for native tools, with all writable homes local."""
    runtime = host_root / ".sandbox"
    home = runtime / "home"
    tmp = runtime / "tmp"
    cache = runtime / "cache"
    config_home = runtime / "config"
    data_home = runtime / "data"
    state_home = runtime / "state"
    clang_cache = cache / "clang"
    swift_cache = cache / "swift"
    for directory in (
        home, tmp, cache, config_home, data_home, state_home, clang_cache, swift_cache
    ):
        directory.mkdir(parents=True, exist_ok=True)

    inherited_names = (
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "COLORTERM",
        "NO_COLOR", "USER", "LOGNAME", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    )
    inherited = {name: os.environ[name] for name in inherited_names if name in os.environ}
    inherited.setdefault(
        "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    )
    return {
        **inherited,
        **_sandbox_env(env, config),
        "ENV_ROOT": str(host_root),
        "HOME": str(home),
        "TMPDIR": f"{tmp}{os.sep}",
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "CFFIXED_USER_HOME": str(home),
        "CLANG_MODULE_CACHE_PATH": str(clang_cache),
        "SWIFT_MODULECACHE_PATH": str(swift_cache),
        "SHELL": "/bin/bash",
    }


class LocalSandbox:
    """Runs commands on the host, rooted at the environment directory."""

    def __init__(self, env: str, config: EnvConfig):
        self.env = env
        self.config = config
        self.host_root = paths.env_dir(env)
        self.env_root = str(self.host_root)

    def start(self) -> None:
        self.host_root.mkdir(parents=True, exist_ok=True)
        _mark_started(self.host_root)

    def stop(self) -> None:
        close_shell_sessions(self.env)
        _mark_stopped(self.host_root)

    def running(self) -> bool:
        return _marked_running(self.host_root)

    def exec(
        self, command: str, cwd: str, timeout: int, session: str | None = None
    ) -> ExecResult:
        self.start()
        workdir = _host_workdir(self.host_root, cwd)
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


class MacOSSandbox:
    """Host-native macOS execution with environment-confined filesystem writes.

    This preserves access to Metal/MLX and host toolchains. It intentionally
    permits reads and network access and should only run trusted agents. The
    Seatbelt boundary prevents the sandboxed process tree from writing or
    deleting files outside this environment and from signalling unrelated
    host processes.
    """

    def __init__(self, env: str, config: EnvConfig):
        self.env = env
        self.config = config
        # Seatbelt matches canonical filesystem paths. In particular, macOS
        # canonicalizes /tmp to /private/tmp.
        self.host_root = paths.env_dir(env).resolve()
        self.env_root = str(self.host_root)

    def _require_seatbelt(self) -> str:
        executable = "/usr/bin/sandbox-exec"
        if sys.platform != "darwin":
            raise RuntimeError("the macos sandbox is only available on macOS")
        if not Path(executable).is_file():
            raise RuntimeError(
                "macOS Seatbelt is unavailable: /usr/bin/sandbox-exec was not found"
            )
        return executable

    def _argv(self, *command: str) -> list[str]:
        darwin_temp, darwin_cache = _darwin_user_write_dirs()
        return [
            self._require_seatbelt(),
            "-D", f"ENV_ROOT={self.env_root}",
            "-D", f"DARWIN_TEMP={darwin_temp}",
            "-D", f"DARWIN_CACHE={darwin_cache}",
            "-D", f"DARWIN_METAL_CACHE={darwin_cache / 'com.apple.metalfe'}",
            "-p", MACOS_SEATBELT_PROFILE,
            *command,
        ]

    def start(self) -> None:
        self._require_seatbelt()
        self.host_root.mkdir(parents=True, exist_ok=True)
        _macos_host_env(self.env, self.config, self.host_root)
        _mark_started(self.host_root)

    def stop(self) -> None:
        close_shell_sessions(self.env)
        _mark_stopped(self.host_root)

    def running(self) -> bool:
        return (
            sys.platform == "darwin"
            and Path("/usr/bin/sandbox-exec").is_file()
            and _marked_running(self.host_root)
        )

    def exec(
        self, command: str, cwd: str, timeout: int, session: str | None = None
    ) -> ExecResult:
        self.start()
        workdir = _host_workdir(self.host_root, cwd)
        environ = _macos_host_env(self.env, self.config, self.host_root)
        if session:
            shell = _persistent_session(
                ("macos", self.env, session),
                lambda: _ShellSession(
                    self._argv("/bin/bash", "--noprofile", "--norc"),
                    workdir,
                    environ,
                ),
            )
            return shell.run(command, timeout)
        try:
            proc = subprocess.run(
                self._argv("/bin/bash", "-lc", command),
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
            raise RuntimeError(
                "docker not found on PATH; use the 'macos' sandbox on macOS "
                "or 'local' for unsafe host execution"
            )
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
    if config.sandbox == "macos":
        return MacOSSandbox(env, config)
    return LocalSandbox(env, config)
