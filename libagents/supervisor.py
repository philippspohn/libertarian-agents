"""Orchestration: who is running, who wakes when.

One thread per active runner, and at most one runner per profile -- enforced
in-process by this registry and across processes by a lock file in the
profile directory. Wake conditions are polled off the board rather than
pushed, so an agent started from the CLI and one started from the UI behave
identically.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import control, paths
from .board import Board
from .runner import Runner

POLL_SECONDS = 1.0


class BudgetLimitError(RuntimeError):
    """The runner cannot resume until an operator raises its absolute cap."""


def budget_limit_reason(env: str, profile: str) -> str | None:
    runner = control.get_runner(env, profile)
    used = control.usage_for(env, profile)
    if used.input_tokens >= runner.config.budgets.input_tokens:
        return (
            f"input budget exhausted ({used.input_tokens}/{runner.config.budgets.input_tokens}); "
            "raise the runner limit before restarting"
        )
    if used.output_tokens >= runner.config.budgets.output_tokens:
        return (
            f"output budget exhausted ({used.output_tokens}/{runner.config.budgets.output_tokens}); "
            "raise the runner limit before restarting"
        )
    env_cap = control.get_env(env).input_token_cap
    env_used = control.usage_for(env).input_tokens
    if env_cap is not None and env_used >= env_cap:
        return (
            f"environment input cap reached ({env_used}/{env_cap}); "
            "raise the environment cap before restarting"
        )
    return None


class ProfileLock:
    """Cross-process guard: at most one runner may instantiate a profile."""

    def __init__(self, env: str, profile: str):
        self.path = paths.profile_dir(env, profile) / ".runner.lock"
        self.acquired = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._stale():
                self.path.unlink(missing_ok=True)
                return self.acquire()
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self.acquired = True
        return True

    def _stale(self) -> bool:
        try:
            pid = int(self.path.read_text().strip())
        except (ValueError, OSError):
            return True
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        return False

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


@dataclass
class Handle:
    env: str
    profile: str
    thread: threading.Thread
    lock: ProfileLock
    stop_flag: threading.Event = field(default_factory=threading.Event)
    wake_flag: threading.Event = field(default_factory=threading.Event)
    wake_reason: str = "started"


class Supervisor:
    def __init__(self) -> None:
        self._handles: dict[tuple[str, str], Handle] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------------------- control

    def is_running(self, env: str, profile: str) -> bool:
        with self._lock:
            h = self._handles.get((env, profile))
        return bool(h and h.thread.is_alive())

    def start(self, env: str, profile: str, reason: str = "started by operator") -> None:
        control.get_runner(env, profile)
        blocked = budget_limit_reason(env, profile)
        if blocked:
            raise BudgetLimitError(blocked)
        with self._lock:
            existing = self._handles.get((env, profile))
            if existing and existing.thread.is_alive():
                raise RuntimeError(f"{env}/{profile} is already running")
            # Acquire synchronously so the caller learns about a runner held by
            # another process instead of silently spawning a thread that dies.
            lock = ProfileLock(env, profile)
            if not lock.acquire():
                raise RuntimeError(
                    f"{env}/{profile} is locked by another process "
                    f"(is `libagents serve` already running it?)"
                )
            handle = Handle(env, profile, thread=None, lock=lock)  # type: ignore[arg-type]
            handle.wake_reason = reason
            thread = threading.Thread(
                target=self._serve, args=(handle,), name=f"runner-{env}-{profile}", daemon=True
            )
            handle.thread = thread
            self._handles[(env, profile)] = handle
        thread.start()

    def stop(self, env: str, profile: str, *, join: float = 0.0) -> bool:
        with self._lock:
            handle = self._handles.get((env, profile))
        if not handle:
            control.set_state(env, profile, "inactive", stop_reason="stopped by operator")
            try:
                Board(paths.board_db(env)).set_status(profile, state="inactive")
            except Exception:
                pass
            return True
        handle.stop_flag.set()
        handle.wake_flag.set()
        if join:
            handle.thread.join(timeout=join)
        return not handle.thread.is_alive()

    def wake(self, env: str, profile: str, reason: str = "woken by operator") -> None:
        blocked = budget_limit_reason(env, profile)
        if blocked:
            raise BudgetLimitError(blocked)
        with self._lock:
            handle = self._handles.get((env, profile))
        if handle and handle.thread.is_alive():
            handle.wake_reason = reason
            handle.wake_flag.set()
            return
        try:
            self.start(env, profile, reason)
        except BudgetLimitError:
            raise
        except RuntimeError:
            pass  # already running, here or in another process

    def stop_env(self, env: str, *, join: float = 0.0) -> list[str]:
        with self._lock:
            handles = [h for (e, _), h in self._handles.items() if e == env]
        for handle in handles:
            handle.stop_flag.set()
            handle.wake_flag.set()
        if join:
            deadline = time.time() + join
            for handle in handles:
                handle.thread.join(timeout=max(0.0, deadline - time.time()))
        remaining = [h.profile for h in handles if h.thread.is_alive()]
        board = Board(paths.board_db(env))
        for runner in control.list_runners(env):
            if runner.profile not in remaining and not self.is_running(env, runner.profile):
                if runner.state != "finished":
                    control.set_state(
                        env, runner.profile, "inactive", stop_reason="environment stopped by operator"
                    )
                    board.set_status(runner.profile, state="inactive")
        return remaining

    def running(self) -> list[tuple[str, str]]:
        with self._lock:
            return [k for k, h in self._handles.items() if h.thread.is_alive()]

    # ------------------------------------------------------------ the thread

    def _serve(self, handle: Handle) -> None:
        env, profile = handle.env, handle.profile
        board = Board(paths.board_db(env))
        reason = handle.wake_reason
        try:
            while not handle.stop_flag.is_set():
                blocked = budget_limit_reason(env, profile)
                if blocked:
                    control.set_state(env, profile, "finished", stop_reason=blocked)
                    board.set_status(profile, status=blocked[:200], state="finished")
                    return
                try:
                    state = Runner(env, profile, stop_flag=handle.stop_flag).run(reason)
                except Exception as exc:
                    control.set_state(env, profile, "inactive", stop_reason=f"runner crashed: {exc}")
                    board.set_status(profile, status=f"crashed: {exc}"[:200], state="inactive")
                    return
                if state != "waiting" or handle.stop_flag.is_set():
                    return
                reason = self._await_wake(handle, board)
                if reason is None:
                    control.set_state(env, profile, "inactive", stop_reason="stopped by operator")
                    board.set_status(profile, state="inactive")
                    return
        finally:
            if handle.stop_flag.is_set():
                try:
                    control.set_state(env, profile, "inactive", stop_reason="stopped by operator")
                    board.set_status(profile, state="inactive")
                except Exception:
                    pass
            handle.lock.release()
            with self._lock:
                if self._handles.get((env, profile)) is handle:
                    self._handles.pop((env, profile), None)

    def _await_wake(self, handle: Handle, board: Board) -> Optional[str]:
        """Block until a message arrives, the timeout expires, or we are
        stopped. Returns the wake reason, or None if stopped."""
        env, profile = handle.env, handle.profile
        while True:
            if handle.stop_flag.is_set():
                return None
            if handle.wake_flag.is_set():
                handle.wake_flag.clear()
                return handle.wake_reason
            try:
                runner = control.get_runner(env, profile)
            except KeyError:
                return None
            if runner.state == "finished":
                return None
            if runner.wake_at and time.time() >= runner.wake_at:
                return "sleep timeout expired"
            n = board.unread_count(profile, after=runner.wake_after_id)
            if n:
                return f"{n} new message(s)"
            time.sleep(POLL_SECONDS)


    # ------------------------------------------------------------ the reaper

    def start_daemon(self, interval: float = 3.0) -> None:
        """Resume runners that are parked as `waiting` but have no live thread
        -- after a server restart, say -- once something would actually have
        woken them. Only a real wake condition (an unread message or an
        expired timer) resumes one, so nothing spends tokens on its own."""
        if getattr(self, "_daemon", None):
            return

        def loop() -> None:
            while True:
                try:
                    self._sweep()
                except Exception:
                    pass
                time.sleep(interval)

        self._daemon = threading.Thread(target=loop, name="supervisor-reaper", daemon=True)
        self._daemon.start()

    def _sweep(self) -> None:
        for env_row in control.list_envs():
            env = env_row["name"]
            board: Optional[Board] = None
            for r in control.list_runners(env):
                if r.state != "waiting" or self.is_running(env, r.profile):
                    continue
                try:
                    if r.wake_at and time.time() >= r.wake_at:
                        self.start(env, r.profile, "sleep timeout expired")
                        continue
                    board = board or Board(paths.board_db(env))
                    if board.unread_count(r.profile, after=r.wake_after_id):
                        self.start(env, r.profile, "new message")
                except RuntimeError:
                    pass  # locked by another process


SUPERVISOR = Supervisor()


def env_status(env: str) -> dict:
    """An environment is 'running' while anyone can still act. Everyone
    asleep without a timeout, finished, or inactive means it is quiescent --
    a message from the operator restarts it."""
    runners = control.list_runners(env)
    live = [r for r in runners if r.state == "active"]
    timed = [r for r in runners if r.state == "waiting" and r.wake_at]
    waiting = [r for r in runners if r.state == "waiting"]
    return {
        "env": env,
        "running": bool(live or timed),
        "active": [r.profile for r in live],
        "waiting": [r.profile for r in waiting],
        "finished": [r.profile for r in runners if r.state == "finished"],
        "inactive": [r.profile for r in runners if r.state == "inactive"],
        "quiescent": not live and not timed,
    }
