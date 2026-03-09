# -*- coding: utf-8 -*-
"""Multi-Agent Pipeline Orchestrator.

Initialises the message bus and all agents, loads :class:`~.agents.TaskContext`
from ``pipeline_config.json``, and manages the iteration loop.

Entry-point integration
-----------------------
``run_pipeline.py`` can launch the orchestrator via::

    from cellscientist.core.orchestrator import run_orchestrator
    asyncio.run(run_orchestrator(config))

Or for a synchronous call::

    from cellscientist.core.orchestrator import run_orchestrator_sync
    run_orchestrator_sync(config)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from .agents import (
    EvaluationAgent,
    ExecutionAgent,
    ModelingAgent,
    ResearchAgent,
    TaskContext,
)
from .message_bus import SimpleMessageBus

logger = logging.getLogger(__name__)


# =============================================================================
# Logging helper (consistent with other core modules)
# =============================================================================


def _log(msg: str, *, console: bool = False) -> None:
    """Print a structured log message.

    Args:
        msg: Message text.
        console: If ``True`` uses ``[CELL_CONSOLE]`` prefix; otherwise ``[DETAIL]``.
    """
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    else:
        print(f"[DETAIL] {msg}", flush=True)


# =============================================================================
# Orchestrator
# =============================================================================


class PipelineOrchestrator:
    """Async orchestrator that coordinates the multi-agent pipeline.

    Initialises a :class:`~.message_bus.SimpleMessageBus`, instantiates one
    agent of each type, and drives the agent loop until the accuracy goal is
    met or ``max_iterations`` is exhausted.

    Attributes:
        config: Full pipeline configuration dict.
        bus: Shared message bus.
        context: Shared task context (SMILES, paths, iteration state).
        research_agent: :class:`~.agents.ResearchAgent` instance.
        modeling_agent: :class:`~.agents.ModelingAgent` instance.
        execution_agent: :class:`~.agents.ExecutionAgent` instance.
        evaluation_agent: :class:`~.agents.EvaluationAgent` instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Create the orchestrator.

        Args:
            config: Full pipeline configuration (``pipeline_config.json``).
        """
        self.config: Dict[str, Any] = config
        self.bus: SimpleMessageBus = SimpleMessageBus()

        # Instantiate agents.
        self.research_agent = ResearchAgent(self.bus, config)
        self.modeling_agent = ModelingAgent(self.bus, config)
        self.execution_agent = ExecutionAgent(self.bus, config)
        self.evaluation_agent = EvaluationAgent(self.bus, config)

        # Build shared task context.
        self.context: TaskContext = TaskContext.from_config(config)

        _log(
            f"[Orchestrator] Initialised task '{self.context.task_id}' "
            f"(max_iterations={self.context.max_iterations}, "
            f"smiles={len(self.context.smiles_list)})",
            console=True,
        )

    async def run(self) -> Dict[str, Any]:
        """Execute the full multi-agent pipeline.

        Kick-off sequence:
        1. Send initial message to :class:`~.agents.ResearchAgent`.
        2. Research → ``biology_insights`` → Modeling
        3. Modeling → ``code_execution`` → Execution
        4. Execution → ``evaluation`` → Evaluation
        5. Evaluation → ``orchestration`` (done) OR feeds back into loop.

        Returns:
            A summary dict with ``status``, ``iterations``, and final
            ``accuracy`` (if detected).
        """
        # Subscribe to the orchestration topic so we can detect completion.
        orchestration_queue = self.bus.subscribe("orchestration")

        # Subscribe each agent to its input topic.
        research_inbox = self.bus.subscribe("biology_insights")
        modeling_inbox = self.bus.subscribe("modeling")
        code_exec_inbox = self.bus.subscribe("code_execution")
        eval_inbox = self.bus.subscribe("evaluation")

        # Launch agent run-loops as background tasks.
        tasks = [
            asyncio.create_task(
                self.research_agent.run(research_inbox), name="research_agent"
            ),
            asyncio.create_task(
                self.modeling_agent.run(modeling_inbox), name="modeling_agent"
            ),
            asyncio.create_task(
                self.execution_agent.run(code_exec_inbox), name="execution_agent"
            ),
            asyncio.create_task(
                self.evaluation_agent.run(eval_inbox), name="evaluation_agent"
            ),
        ]

        # Seed the pipeline.
        initial_message = {
            "smiles_list": self.context.smiles_list,
            "h5_file_path": self.context.h5_file_path,
            "context_text": "",
            "stage": "design",
            "iteration": 0,
            "max_iterations": self.context.max_iterations,
        }
        # The ResearchAgent subscribes to biology_insights; trigger it.
        await self.bus.publish("biology_insights", initial_message)

        # Wait for the pipeline to signal completion via orchestration topic.
        pipeline_timeout: float = float(
            (self.config.get("exec") or {}).get("timeout_seconds") or 7200
        )
        _log("[Orchestrator] Pipeline started; waiting for completion…", console=True)
        result: Optional[Dict[str, Any]] = None
        try:
            result = await asyncio.wait_for(
                orchestration_queue.get(),
                timeout=pipeline_timeout,
            )
        except asyncio.TimeoutError:
            _log(
                f"[Orchestrator] ⚠️ Pipeline timed out after {pipeline_timeout:.0f}s "
                "before reaching orchestration topic.",
                console=True,
            )
            result = {"status": "timeout", "data": {}}
        finally:
            # Cancel all agent background tasks.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        status = (result or {}).get("status", "unknown")
        data = (result or {}).get("data") or {}
        accuracy = data.get("accuracy")
        iterations_done = self.context.iteration

        summary = {
            "status": status,
            "task_id": self.context.task_id,
            "iterations": iterations_done,
            "accuracy": accuracy,
            "max_iterations_reached": data.get("max_iterations_reached", False),
        }

        _log(
            f"[Orchestrator] Pipeline finished: status={status}, "
            f"accuracy={accuracy}, iterations={iterations_done}",
            console=True,
        )
        return summary


# =============================================================================
# Public helpers
# =============================================================================


async def run_orchestrator(config: Dict[str, Any]) -> Dict[str, Any]:
    """Async entry-point: create the orchestrator and run the pipeline.

    Args:
        config: Full pipeline configuration dict.

    Returns:
        Summary dict from :meth:`PipelineOrchestrator.run`.
    """
    orchestrator = PipelineOrchestrator(config)
    return await orchestrator.run()


def run_orchestrator_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper around :func:`run_orchestrator`.

    Convenience function for callers that cannot use ``await``.

    Args:
        config: Full pipeline configuration dict.

    Returns:
        Summary dict from :meth:`PipelineOrchestrator.run`.
    """
    return asyncio.run(run_orchestrator(config))


def load_config(pipeline_config_path: str = "configs/pipeline_config.json") -> Dict[str, Any]:
    """Load and return the pipeline configuration from a JSON file.

    Args:
        pipeline_config_path: Path to ``pipeline_config.json``.

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    abs_path = os.path.abspath(pipeline_config_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Pipeline config not found at: {abs_path}"
        )
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)
