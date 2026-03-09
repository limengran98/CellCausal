# -*- coding: utf-8 -*-
"""Structured logging for the CellCausal multi-agent pipeline.

Provides :class:`StructuredLogger`, which emits log lines in the standard
format::

    [{timestamp}] | [{agent_name}] | [{state}/{legacy_phase}] | [{message}]

Example::

    [2026-03-09T10:30:00Z] | [ModelingAgent] | [MODEL_GENERATION/Phase: Experiment] | [Generating PyTorch code]

The logger outputs to the console (via the existing ``_log``-style
``[CELL_CONSOLE]`` / ``[DETAIL]`` prefixes) **and** to the Python
:mod:`logging` infrastructure so that any attached :class:`~logging.Handler`
(e.g., the existing ``TieredLogger``) also receives the messages.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _log(msg: str, *, console: bool = False) -> None:
    """Emit a message using the standard CellCausal prefix convention.

    Args:
        msg: Message text.
        console: If ``True`` uses ``[CELL_CONSOLE]`` prefix; otherwise ``[DETAIL]``.
    """
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    else:
        print(f"[DETAIL] {msg}", flush=True)


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format (``YYYY-MM-DDTHH:MM:SSZ``)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StructuredLogger:
    """Emit structured log lines for a named component.

    Each log line follows the format::

        [{timestamp}] | [{agent_name}] | [{state}/{legacy_phase}] | [{message}]

    The *state* argument passed to each method should be a plain string such as
    ``"MODEL_GENERATION/Phase: Experiment"`` or simply a
    :class:`~cellscientist.core.orchestrator.PipelineState` enum value whose
    ``str()`` already contains the legacy alias (the orchestrator formats
    these for you when transitioning states).

    Output goes to:
    1. ``stdout`` via the legacy ``[CELL_CONSOLE]`` / ``[DETAIL]`` prefix so
       that existing monitoring/log-parsers keep working.
    2. Python :mod:`logging` at the corresponding level so that any attached
       ``TieredLogger`` handler also receives the message.

    Example::

        sl = StructuredLogger("ModelingAgent")
        sl.info("MODEL_GENERATION/Phase: Experiment", "Generating PyTorch code")
    """

    def __init__(self, name: str, *, console: bool = True) -> None:
        """Initialise the logger.

        Args:
            name: Human-readable component name (e.g. ``"ModelingAgent"``).
            console: When ``True`` (default) messages are printed to stdout
                with the ``[CELL_CONSOLE]`` prefix in addition to being sent
                to the Python logging system.
        """
        self.name: str = name
        self._console: bool = console
        self._py_logger = logging.getLogger(f"cellscientist.{name}")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def info(self, agent_name: str, state: str, message: str) -> None:
        """Log an informational message.

        Args:
            agent_name: Name of the emitting agent or component.
            state: Current FSM state and/or legacy phase string.
            message: Human-readable log message.
        """
        self._emit(logging.INFO, agent_name, state, message)

    def warning(self, agent_name: str, state: str, message: str) -> None:
        """Log a warning message.

        Args:
            agent_name: Name of the emitting agent or component.
            state: Current FSM state and/or legacy phase string.
            message: Human-readable log message.
        """
        self._emit(logging.WARNING, agent_name, state, message)

    def error(self, agent_name: str, state: str, message: str) -> None:
        """Log an error message.

        Args:
            agent_name: Name of the emitting agent or component.
            state: Current FSM state and/or legacy phase string.
            message: Human-readable log message.
        """
        self._emit(logging.ERROR, agent_name, state, message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, level: int, agent_name: str, state: str, message: str) -> None:
        """Format and emit the structured log line.

        Args:
            level: Python :mod:`logging` level constant.
            agent_name: Emitting agent name.
            state: FSM state / legacy phase label.
            message: Log message body.
        """
        line = f"[{_now_iso()}] | [{agent_name}] | [{state}] | [{message}]"

        # Console output (uses existing [CELL_CONSOLE] / [DETAIL] convention).
        _log(line, console=self._console)

        # Python logging infrastructure.
        self._py_logger.log(level, line)


# Module-level default instance for convenience when a single shared logger
# is acceptable (e.g. inside agent methods that don't construct their own).
_default_structured_logger: Optional[StructuredLogger] = None


def get_default_structured_logger(name: str = "Pipeline") -> StructuredLogger:
    """Return (or create) a module-level :class:`StructuredLogger` singleton.

    Args:
        name: Name used when creating a new instance.

    Returns:
        Shared :class:`StructuredLogger` instance.
    """
    global _default_structured_logger
    if _default_structured_logger is None:
        _default_structured_logger = StructuredLogger(name)
    return _default_structured_logger
