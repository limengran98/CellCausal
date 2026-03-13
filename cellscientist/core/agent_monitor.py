# -*- coding: utf-8 -*-
"""Real-time agent monitoring: ``@monitor_agent`` decorator and :class:`Heartbeat`.

This module provides lightweight observability tooling for the Multi-Agent
Pipeline:

* :func:`monitor_agent` — decorator that wraps an agent's ``process()`` method
  with start/stop timing and color-coded console output.
* :class:`Heartbeat` — async context manager that emits a periodic status
  message while an agent is working, so operators can confirm the pipeline has
  not stalled.

Usage example::

    from cellscientist.core.agent_monitor import monitor_agent, Heartbeat

    class MyAgent(BaseAgent):
        @monitor_agent
        async def process(self, message):
            async with Heartbeat(self.role):
                ...
"""

from __future__ import annotations

import os

import asyncio
import functools
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# =============================================================================
# Logging helper (consistent with other core modules)
# =============================================================================


def _log(msg: str, *, console: bool = False) -> None:
    """Print a structured log message using the standard CellCausal prefix.

    Args:
        msg: Message text.
        console: If ``True`` uses ``[CELL_CONSOLE]`` prefix so the message
            appears in the console output; otherwise uses ``[DETAIL]``.
    """
    summary_only = str(os.environ.get("CELL_SUMMARY_ONLY", "0")).lower() in {"1", "true", "yes"}
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    elif not summary_only:
        print(f"[DETAIL] {msg}", flush=True)


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# @monitor_agent decorator
# =============================================================================


def monitor_agent(func: Callable) -> Callable:
    """Decorate an agent ``process()`` coroutine with timing and status output.

    Before execution, prints::

        [AGENT] 🚀 {agent_role} starting task... (2024-01-01T00:00:00Z)

    On success, prints::

        [AGENT] ✅ {agent_role} completed in 3.2s

    On failure, prints::

        [AGENT] ❌ {agent_role} failed: <error message>

    The decorator is transparent: it preserves the function signature and
    propagates exceptions unchanged.

    Args:
        func: An async ``process(self, message, ...)`` method to wrap.

    Returns:
        Wrapped coroutine function with the same signature.
    """

    @functools.wraps(func)
    async def wrapper(self: Any, message: dict, *args: Any, **kwargs: Any) -> Any:
        agent_role: str = getattr(self, "role", self.__class__.__name__)
        ts = _now_iso()
        _log(f"[AGENT] {agent_role} START", console=True)
        _log(f"├─ Time: {ts}", console=True)
        if isinstance(message, dict):
            fold = message.get("fold") or message.get("fold_id")
            scope = message.get("scope") or message.get("split")
            if fold is not None or scope is not None:
                fold_text = fold if fold is not None else 'N/A'
                scope_text = scope if scope is not None else 'N/A'
                _log(f"├─ Fold: {fold_text} | Scope: {scope_text}", console=True)

        start = time.monotonic()
        try:
            result = await func(self, message, *args, **kwargs)
            duration = time.monotonic() - start
            _log(f"└─ Duration: {duration:.1f}s", console=True)
            _log(f"[AGENT] {agent_role} END ✅", console=True)
            return result
        except Exception as exc:
            duration = time.monotonic() - start
            _log(
                f"└─ Duration: {duration:.1f}s",
                console=True,
            )
            _log(f"[AGENT] {agent_role} END ❌: {exc}", console=True)
            raise

    return wrapper


# =============================================================================
# Heartbeat async context manager
# =============================================================================


class Heartbeat:
    """Async context manager that emits periodic status messages.

    Useful inside long-running ``process()`` methods to prevent silent stalls.
    The heartbeat runs as a background :class:`asyncio.Task` and is
    automatically cancelled when the context exits.

    Example::

        async with Heartbeat(self.role, interval=10.0):
            result = await some_long_operation()

    Args:
        agent_role: Human-readable label for the running agent.
        interval: Seconds between each heartbeat message (default ``10.0``).
    """

    def __init__(self, agent_role: str, interval: float = 10.0) -> None:
        self.agent_role = agent_role
        self.interval = interval
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._start_time: float = 0.0

    async def __aenter__(self) -> "Heartbeat":
        self._start_time = time.monotonic()
        self._task = asyncio.create_task(self._beat())
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _beat(self) -> None:
        """Background coroutine that prints a status line every *interval* s."""
        while True:
            await asyncio.sleep(self.interval)
            elapsed = time.monotonic() - self._start_time
            _log(
                f"[HEARTBEAT] 💓 {self.agent_role} still working... ({elapsed:.0f}s)",
                console=True,
            )
