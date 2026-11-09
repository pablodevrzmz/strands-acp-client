"""The event loop every agent process of one tool lives on, and the bridges into and out of it."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import logging
import sys
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from typing import Any, TypeVar

__all__ = ["CallCancelled", "CallerGone", "Runtime", "RuntimeClosed", "call_on_loop", "unless_cancelled"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

_POLL = 0.2
"""Seconds between two looks at a caller's cancellation signal, which is a ``threading.Event``."""


class RuntimeClosed(RuntimeError):
    """The runtime was stopped while a call was running on it."""


class CallerGone(RuntimeError):
    """The event loop a call came from has gone away, so nothing can run on it any more."""


class CallCancelled(Exception):
    """The caller's cancellation signal was set while the call was waiting."""


class Runtime:
    """One event loop on one daemon thread, owned by one tool.

    An agent subprocess and its ACP connection belong to the event loop that created them.
    Strands runs every synchronous ``agent(...)`` call on a loop of its own and closes that loop
    when the call returns, so a process started there could not be reached by the next call. Here
    the processes live on a loop that belongs to the tool instead: calls arriving from any loop or
    thread reach the same processes, and a ``session_id`` works for as long as the tool does.

    The thread starts on first use and again after :meth:`stop`, so a tool that is never called
    costs nothing and a closed tool can be used again.
    """

    def __init__(self, name: str = "strands-acp-client") -> None:
        self._name = name
        # Reentrant: the garbage collector may run ``abandon`` on a thread that is inside ``_submit``.
        self._guard = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Put ``coro`` on the loop, starting the loop and its thread if they are not running.

        Under the guard, so that ``stop`` cannot retire the loop between the moment it is chosen
        and the moment the coroutine is on it. Whatever is submitted is then either run or
        cancelled, and never left waiting on a loop that has already finished.
        """
        with self._guard:
            loop, thread = self._loop, self._thread
            if loop is None or thread is None or not thread.is_alive():
                loop = _new_event_loop()
                thread = threading.Thread(target=_serve, args=(loop,), name=self._name, daemon=True)
                thread.start()
                self._loop, self._thread = loop, thread
            return _schedule(loop, coro)

    def _is_current(self) -> bool:
        """Whether the caller is already running on the runtime loop."""
        loop = self._loop
        return loop is not None and asyncio.get_running_loop() is loop

    # ── from the caller's loop into this one ──

    async def call(self, func: Callable[..., Coroutine[Any, Any, T]], *args: Any) -> T:
        """Await ``func(*args)`` on the runtime loop, from whichever loop the caller is on."""
        if self._is_current():
            return await func(*args)
        return await _relay(self._submit(func(*args)), RuntimeClosed("the tool was closed while the call was running"))

    async def stream(self, make: Callable[[], AsyncGenerator[T, None]]) -> AsyncGenerator[T, None]:
        """Iterate ``make()`` on the runtime loop and hand each item to the caller's loop.

        When the caller stops listening (it is cancelled, it closes this generator early, or its
        loop goes away) the iteration is cancelled on the runtime side too. Its ``finally`` blocks
        still run to the end, because the loop they need is this one and not the caller's.
        """
        caller = asyncio.get_running_loop()
        if self._is_current():
            async with contextlib.aclosing(make()) as items:
                async for item in items:
                    yield item
            return

        inbox: asyncio.Queue[tuple[bool, Any]] = asyncio.Queue()

        def post(last: bool, value: Any) -> bool:
            try:
                caller.call_soon_threadsafe(inbox.put_nowait, (last, value))
            except RuntimeError:  # the caller's loop is closed: nobody is listening any more
                return False
            return True

        async def pump() -> None:
            outcome: BaseException | None = None
            try:
                async with contextlib.aclosing(make()) as items:
                    async for item in items:
                        if not post(False, item):
                            return
            except asyncio.CancelledError:
                # Cancelled by a caller that left, who will never read this, or by ``stop`` under a
                # caller that is still waiting and must be told why no answer is coming.
                outcome = RuntimeClosed("the tool was closed while the call was running")
                raise
            except Exception as exc:
                outcome = exc
            finally:
                post(True, outcome)

        future = self._submit(pump())
        finished = False
        try:
            while True:
                last, value = await inbox.get()
                if last:
                    finished = True
                    if value is not None:
                        raise value
                    return
                yield value
        finally:
            if not finished:
                future.cancel()

    # ── ending it ──

    def stop(self, cleanup: Callable[[], Coroutine[Any, Any, object]] | None = None, timeout: float = 15.0) -> None:
        """Run ``cleanup`` on the loop, then stop the loop and wait for its thread.

        Blocks the calling thread, so it cannot be called from the runtime loop itself. Calls still
        running are cancelled and their callers get :class:`RuntimeClosed`. Stopping a runtime
        that never started does nothing.
        """
        with self._guard:
            loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        if thread is threading.current_thread():
            raise RuntimeError("the runtime cannot be stopped from its own loop")
        if cleanup is not None and thread.is_alive():
            try:
                _schedule(loop, cleanup()).result(timeout)
            except Exception:
                logger.warning("closing the agent processes did not finish cleanly", exc_info=True)
        with self._guard:
            # From here on a new call starts a new loop. Every call already on this one was put
            # there before the stop below, so it is cancelled and answered rather than lost.
            if self._loop is loop:
                self._loop = self._thread = None
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("the %s thread did not stop within %.0fs", self._name, timeout)

    def abandon(self, cleanup: Callable[[], Coroutine[Any, Any, object]]) -> None:
        """Stop without waiting, for a tool that was dropped without being closed.

        This runs wherever the garbage collector happens to be, so it must not block: the loop
        finishes ``cleanup`` on its own thread and then stops itself.
        """
        with self._guard:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None or thread is None or not thread.is_alive():
            return

        async def finish() -> None:
            try:
                await cleanup()
            finally:
                loop.stop()

        with contextlib.suppress(RuntimeClosed):
            _schedule(loop, finish())

    def forget(self) -> None:
        """Drop the loop without touching it: in a forked child it belongs to the parent process."""
        self._guard = threading.RLock()
        if self._loop is not None:
            _ORPHANED_LOOPS.append(self._loop)  # never finalised here: its file descriptors are the parent's
        self._loop = self._thread = None


_ORPHANED_LOOPS: list[asyncio.AbstractEventLoop] = []


def _new_event_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        # The one loop on Windows that can run subprocesses, whatever policy the application has set.
        return asyncio.ProactorEventLoop()
    # Elsewhere the application's policy decides. Before Python 3.12 a loop can only watch its
    # child processes through the policy it came from, and an application on uvloop gets a uvloop
    # loop here, which runs subprocesses by itself.
    return asyncio.new_event_loop()


def _serve(loop: asyncio.AbstractEventLoop) -> None:
    try:
        loop.run_forever()
    finally:
        try:
            # Cancel what is still running and wait for it to unwind, as ``asyncio.run`` does. A
            # second round catches what the first one started and nobody waited for.
            for _ in range(8):
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                if not pending:
                    break
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            logger.debug("the runtime loop did not shut down cleanly", exc_info=True)
        finally:
            loop.close()


def _schedule(loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:  # the loop is closed
        coro.close()
        raise RuntimeClosed("the tool was closed") from None


async def _relay(source: concurrent.futures.Future[T], gone: Exception) -> T:
    """Await a future that is resolved on another loop.

    ``asyncio.wrap_future`` would turn a cancellation on the other side into ``CancelledError``
    here, which reads as "this task was cancelled" and is wrong: this side did nothing. So the
    other side going away raises ``gone`` instead, and only a real cancellation of this task is
    passed on to the other side.
    """
    here = asyncio.get_running_loop()
    mine: asyncio.Future[T] = here.create_future()

    def settle() -> None:
        if mine.done():
            return
        if source.cancelled():
            mine.set_exception(gone)
            return
        failure = source.exception()
        if failure is not None:
            mine.set_exception(failure)
        else:
            mine.set_result(source.result())

    def on_done(_: concurrent.futures.Future[T]) -> None:
        with contextlib.suppress(RuntimeError):  # this loop is closed: nobody is waiting any more
            here.call_soon_threadsafe(settle)

    source.add_done_callback(on_done)
    try:
        return await mine
    except asyncio.CancelledError:
        source.cancel()
        raise


# ── from this loop back to the caller's ──


async def call_on_loop(home: asyncio.AbstractEventLoop | None, func: Callable[..., Any], *args: Any) -> Any:
    """Call ``func(*args)`` on ``home`` and await there whatever it returns.

    This is how an application's permission handler keeps running where the application expects
    it, on the event loop of the call that started the turn, although the agent it answers lives
    on the tool's own loop.
    """

    async def invoke() -> Any:
        value = func(*args)
        return await value if inspect.isawaitable(value) else value

    if home is None or home is asyncio.get_running_loop():
        return await invoke()
    coro = invoke()
    try:
        source = asyncio.run_coroutine_threadsafe(coro, home)
    except RuntimeError:
        coro.close()
        raise CallerGone("the event loop that started this call is closed") from None
    return await _relay(source, CallerGone("the call that started this turn went away"))


async def unless_cancelled(awaitable: Awaitable[T], cancel_signal: threading.Event | None) -> T:
    """Await ``awaitable``, giving up with :class:`CallCancelled` once ``cancel_signal`` is set."""
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=None if cancel_signal is None else _POLL)
            if done:
                return task.result()
            if cancel_signal is not None and cancel_signal.is_set():
                raise CallCancelled
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
