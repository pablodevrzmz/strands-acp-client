"""The live sessions of one tool, and the rules for reusing, resuming and evicting them."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import NamedTuple

from strands_acp_client._runtime import Runtime
from strands_acp_client._session import AcpSession, SessionSpec

__all__ = ["SessionOwner", "SessionRegistry"]

logger = logging.getLogger(__name__)

_REMEMBERED = 1024
"""How many session ids are remembered with their agent and directory after their process is gone."""


_PARENTS_SESSIONS: list[AcpSession] = []
"""In a forked child: the sessions it inherited, which belong to the parent process."""


class SessionOwner(NamedTuple):
    """Which agent a session belongs to and where it works."""

    agent_name: str
    cwd: str


class SessionRegistry:
    """Live ACP sessions of one tool, keyed by session id.

    Reachable as ``tool.acp_sessions``. ``len()``, :attr:`session_ids`, :meth:`close` and
    :meth:`close_all` can be used from any thread or event loop. The agent processes themselves
    live on the tool's own loop, and so does everything else in this class.
    """

    def __init__(self, runtime: Runtime, max_sessions: int = 8) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self.max_sessions = max_sessions
        self._runtime = runtime
        self._sessions: dict[str, AcpSession] = {}
        self._opening: dict[str, asyncio.Future[None]] = {}
        self._owners: OrderedDict[str, SessionOwner] = OrderedDict()

    # ── for the application: any thread, any loop ──

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def session_ids(self) -> list[str]:
        """Ids of the sessions whose agent process is alive."""
        return list(self._sessions)

    async def close(self, session_id: str) -> None:
        """End the agent process holding ``session_id``, if there is one."""
        if session_id in self._sessions:
            await self._runtime.call(self.drop_id, session_id)

    async def close_all(self) -> None:
        """End every agent process. The tool stays usable and starts new ones on demand."""
        if self._sessions:
            await self._runtime.call(self.drop_all)

    # ── for the tool: on its own loop ──

    def owner(self, session_id: str) -> SessionOwner | None:
        """The agent and directory of a session this tool has seen, alive or not."""
        held = self._sessions.get(session_id)
        if held is not None:
            return SessionOwner(held.agent_name, held.cwd)
        return self._owners.get(session_id)

    async def acquire(self, spec: SessionSpec, session_id: str | None = None) -> AcpSession:
        """Lease the live session with this id, or start a process for a new or resumed one.

        A leased session is never evicted. Every ``acquire`` is paired with a :meth:`release`.
        """
        if session_id is None:
            return await self._start(spec, None)
        while True:
            held = self._sessions.get(session_id)
            if held is not None and held.alive:
                return self._lease(held)
            if held is not None:
                await self._drop(held)  # its process is gone
            opening = self._opening.get(session_id)
            if opening is None:
                break
            # Another call is already resuming this session. Two processes must not load the
            # same session, so wait for that one and look again.
            await asyncio.wait({opening})
        marker: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._opening[session_id] = marker
        try:
            return await self._start(spec, session_id)
        finally:
            del self._opening[session_id]
            marker.set_result(None)

    async def release(self, session: AcpSession, keep: bool = True) -> None:
        """Give a lease back. ``keep=False`` closes the session once nobody else holds it."""
        session.leases = max(session.leases - 1, 0)
        session.touch()
        if not keep:
            session.retired = True
        if not session.alive or (session.retired and session.leases == 0):
            await self._drop(session)
        await self._evict()

    def holds(self, session: AcpSession) -> bool:
        """Whether a follow-up with this session's id would reach this very process."""
        return self._sessions.get(session.session_id) is session and session.alive and not session.retired

    async def drop_id(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            await self._drop(session)

    async def drop_all(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(self._close(session) for session in sessions))

    def kill_all(self) -> None:
        """End every process at once, without the loop. For when the loop is already gone."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            session.kill()

    def forget(self) -> None:
        """Drop every session without touching its process: in a forked child they are the parent's.

        The session objects are kept, never to be used again. Letting them be collected would run
        the finalizers of their subprocess transports, which close what they still take for theirs.
        """
        _PARENTS_SESSIONS.extend(self._sessions.values())
        self._sessions = {}
        self._opening = {}

    async def _start(self, spec: SessionSpec, session_id: str | None) -> AcpSession:
        session = AcpSession(spec)
        await session.start(session_id)
        try:
            previous = self._sessions.get(session.session_id)
            if previous is not None:
                # Two agents handing out the same id. One id means one process here, so the older goes.
                logger.warning("session id %r was issued twice; closing the earlier session", session.session_id)
                await self._drop(previous)
            self._sessions[session.session_id] = session
            self._owners[session.session_id] = SessionOwner(session.agent_name, session.cwd)
            self._owners.move_to_end(session.session_id)
            while len(self._owners) > _REMEMBERED:
                self._owners.popitem(last=False)
            self._lease(session)
            await self._evict()
        except BaseException:
            # Cancelled while making room. The caller never sees this session, so nobody could give
            # its lease back or name it again.
            session.leases = 0
            await self._drop(session)
            raise
        return session

    def _lease(self, session: AcpSession) -> AcpSession:
        session.leases += 1
        session.touch()
        return session

    async def _drop(self, session: AcpSession) -> None:
        if self._sessions.get(session.session_id) is session:
            del self._sessions[session.session_id]
        await self._close(session)

    async def _close(self, session: AcpSession) -> None:
        try:
            await session.close()
        except Exception:
            logger.warning("could not close session %r cleanly; killing its process", session.session_id, exc_info=True)
            session.kill()

    async def _evict(self) -> None:
        """Close idle sessions, least recently used first, until the limit holds again.

        Sessions in use are never closed. When all of them are, the count stays above the limit
        until a call finishes and its :meth:`release` runs this again.
        """
        while len(self._sessions) > self.max_sessions:
            idle = [session for session in self._sessions.values() if session.leases == 0]
            if not idle:
                return
            await self._drop(min(idle, key=lambda session: session.last_used))
