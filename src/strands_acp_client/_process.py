"""Starting an agent as a subprocess, hearing what it says on stderr, and making sure it ends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import sys
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["EXIT_GRACE", "AgentProcess"]

logger = logging.getLogger(__name__)

EXIT_GRACE = 2.0
"""Seconds an agent gets at each step of a shutdown: after its stdin closes, then after SIGTERM."""

_TAIL_BYTES = 8192
_TAIL_LINES = 12
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# The agent leads a process group of its own, so that one signal reaches it and whatever it started.
_OWN_GROUP: dict[str, Any] = {} if sys.platform == "win32" else {"start_new_session": True}


class AgentProcess:
    """One agent subprocess: its pipes, its last words on stderr, and how it is stopped.

    Agents are often launched through a wrapper (``npx`` starts ``node``, which is the agent).
    Signalling the wrapper alone can leave the real agent running, so the process is started as
    the leader of its own process group and signals go to the group. On Windows there is no
    such group and only the process itself is terminated.

    Its stderr is read for as long as it runs. Left unread, the pipe fills up (64 KiB on Linux)
    and an agent that logs there blocks for good in the middle of a turn. The last lines are
    kept, because that is where an agent says why it would not start.
    """

    def __init__(self, proc: asyncio.subprocess.Process, label: str) -> None:
        self._proc = proc
        self._label = label
        self._tail = bytearray()
        self._stderr = asyncio.ensure_future(self._read_stderr())

    @classmethod
    async def spawn(cls, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], label: str) -> AgentProcess:
        """Start ``argv`` with its three standard streams as pipes."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=dict(env),
            **_OWN_GROUP,
        )
        return cls(proc, label)

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    @property
    def stdin(self) -> asyncio.StreamWriter:
        assert self._proc.stdin is not None, "the process was spawned with a stdin pipe"
        return self._proc.stdin

    @property
    def stdout(self) -> asyncio.StreamReader:
        assert self._proc.stdout is not None, "the process was spawned with a stdout pipe"
        return self._proc.stdout

    # ── what it says ──

    async def _read_stderr(self) -> None:
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            while chunk := await stream.read(65536):
                self._tail += chunk
                del self._tail[:-_TAIL_BYTES]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("%s (pid %d) stderr: %s", self._label, self._proc.pid, _clean(chunk).rstrip())
        except Exception:
            logger.debug("stopped reading stderr of %s (pid %d)", self._label, self._proc.pid, exc_info=True)

    def stderr_tail(self) -> str:
        """The last lines the agent wrote to stderr, without colour codes."""
        lines = [line.rstrip() for line in _clean(bytes(self._tail)).splitlines() if line.strip()]
        return "\n".join(lines[-_TAIL_LINES:])

    def exit_note(self) -> str:
        """How the process ended, as a clause for an error message; empty while it is running."""
        code = self._proc.returncode
        if code is None:
            return ""
        if code < 0:
            return f"the process was killed by {_signal_name(-code)}"
        return f"the process exited with code {code}"

    async def last_words(self, wait: float = 1.0) -> tuple[str, str]:
        """How the process ended and what it last wrote to stderr.

        A process that is on its way out gets ``wait`` seconds to finish, so that the exit code
        and the end of its stderr are there to be quoted.
        """
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._proc.wait(), wait)
        if self._proc.returncode is not None and wait > 0:
            await asyncio.wait({self._stderr}, timeout=wait)
        return self.exit_note(), self.stderr_tail()

    # ── how it ends ──

    async def stop(self) -> None:
        """End the process: let it exit on its own, then SIGTERM its group, then SIGKILL its group."""
        if self._proc.returncode is None:
            try:
                with contextlib.suppress(Exception):
                    self.stdin.close()  # end of input is how an agent on stdio is told to leave
                if not await self._exits(EXIT_GRACE):
                    logger.debug("%s (pid %d) is still running after its stdin closed", self._label, self.pid)
                    self._signal_group(signal.SIGTERM)
                    if not await self._exits(EXIT_GRACE):
                        logger.warning("agent %s (pid %d) ignored SIGTERM; killing it", self._label, self.pid)
            finally:
                # Always: this ends an agent that ignored SIGTERM, sweeps up what a well-behaved one
                # left behind in its group, and leaves no orphan if this coroutine is cancelled half
                # way. The leader was alive when this began, so the group id is still its own.
                self._kill_group()
            await self._exits(EXIT_GRACE)
        if not self._stderr.done():
            await asyncio.wait({self._stderr}, timeout=0.5)
            self._stderr.cancel()

    def kill(self) -> None:
        """Kill the agent and its process group at once. Synchronous: it needs no event loop.

        Nothing is signalled once the process is known to have exited. Its id may have been given
        to an unrelated process since, and a group is addressed by that id.
        """
        if self._proc.returncode is None:
            self._kill_group()

    async def _exits(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._proc.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    def _signal_group(self, sig: int) -> None:
        if sys.platform == "win32":
            with contextlib.suppress(ProcessLookupError, OSError):
                self._proc.terminate()
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._proc.pid, sig)

    def _kill_group(self) -> None:
        if sys.platform == "win32":
            with contextlib.suppress(ProcessLookupError, OSError):
                self._proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._proc.pid, signal.SIGKILL)


def _clean(data: bytes) -> str:
    return _ANSI.sub("", data.decode("utf-8", errors="replace"))


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"
