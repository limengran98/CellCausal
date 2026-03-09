# -*- coding: utf-8 -*-
"""Centralised prompt loading and formatting for all agents.

Loads ``configs/prompts.yaml`` once and exposes a simple
:meth:`PromptManager.get_prompt` interface.  Falls back to hardcoded
defaults when the YAML file is missing or a key is absent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallbacks (kept for backward compatibility when YAML is absent)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict[str, str]] = {
    "research_agent": {
        "system_prompt": (
            "You are a biological researcher specialising in cell perturbation biology. "
            "Analyse the provided SMILES compounds and contextual data to generate a "
            "comprehensive Biological Insight Report."
        ),
        "analysis_template": (
            "Analyse the following SMILES compounds and generate a Biological Insight Report:\n\n"
            "## SMILES Compounds\n{smiles_list}\n\n"
            "## Context\n{context_text}\n\n"
            "## Literature Summary\n{literature_summary}"
        ),
    },
    "modeling_agent": {
        "system_prompt": (
            "You are an expert PyTorch/GNN engineer specialising in cell perturbation modeling. "
            "Write complete, runnable Python code for a GNN model that predicts cell painting "
            "perturbation responses. Return ONLY the Python code."
        ),
        "code_generation_template": (
            "## Biological Context\n{literature}\n\n"
            "## SMILES Compounds (sample)\n{smiles_list}"
        ),
        "self_correction_system_prompt": (
            "You are an expert PyTorch/GNN engineer. "
            "The code below raised errors. Fix ALL errors and return ONLY the corrected Python code."
        ),
        "self_correction_template": (
            "## Error Logs\n{error_logs}\n\n"
            "## Existing Code\n```python\n{existing_code}\n```"
        ),
    },
    "execution_agent": {
        "system_prompt": (
            "You are a code executor responsible for running generated PyTorch/GNN code "
            "in a sandboxed environment and capturing all output for downstream evaluation."
        ),
    },
    "evaluation_agent": {
        "system_prompt": (
            "You are an evaluator assessing model performance against biological accuracy goals."
        ),
        "feedback_template": (
            "Compare results against 90% accuracy goal.\n\n"
            "Current accuracy: {accuracy}\n"
            "Target: 90%\n\n"
            "Please provide deeper biological insights to guide model improvement."
        ),
    },
}


class PromptManager:
    """Load and format agent prompt templates from ``configs/prompts.yaml``.

    The YAML file is loaded once and cached in memory.  If the file is
    missing or a requested key does not exist the manager falls back to
    the hardcoded :data:`_DEFAULTS`.

    Example::

        pm = PromptManager()
        system_msg = pm.get_prompt("modeling_agent", "system_prompt")
        user_msg = pm.get_prompt(
            "modeling_agent",
            "code_generation_template",
            literature="...",
            smiles_list="...",
        )
    """

    def __init__(self, prompts_yaml_path: Optional[Path] = None) -> None:
        """Initialise and load the YAML prompt file.

        Args:
            prompts_yaml_path: Optional explicit path to ``prompts.yaml``.
                Defaults to ``<project_root>/configs/prompts.yaml``.
        """
        if prompts_yaml_path is None:
            # Resolve relative to this file's package tree.
            prompts_yaml_path = (
                Path(__file__).resolve().parent.parent.parent
                / "configs"
                / "prompts.yaml"
            )
        self._path: Path = prompts_yaml_path
        self._cache: Dict[str, Any] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prompt(self, agent_name: str, template_name: str, **kwargs: Any) -> str:
        """Return a (optionally formatted) prompt template.

        Args:
            agent_name: Top-level YAML key, e.g. ``"modeling_agent"``.
            template_name: Second-level YAML key, e.g. ``"system_prompt"``.
            **kwargs: Format arguments applied to the template via
                :meth:`str.format_map`.  Missing keys are silently ignored
                so partially-populated templates are still returned.

        Returns:
            The formatted prompt string.  Falls back to hardcoded defaults
            if the YAML file or key is unavailable.
        """
        self._ensure_loaded()
        raw = self._lookup(agent_name, template_name)
        if not raw:
            return ""
        if kwargs:
            try:
                return raw.format_map(_SafeFormatMap(kwargs))
            except Exception as exc:
                logger.debug(
                    "[PromptManager] Template formatting error for %s/%s: %s",
                    agent_name,
                    template_name,
                    exc,
                )
                return raw
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the YAML file if not already done."""
        if self._loaded:
            return
        self._loaded = True
        try:
            import yaml  # type: ignore

            with open(self._path, "r", encoding="utf-8") as fh:
                self._cache = yaml.safe_load(fh) or {}
            logger.debug("[PromptManager] Loaded prompts from '%s'.", self._path)
        except FileNotFoundError:
            logger.warning(
                "[PromptManager] prompts.yaml not found at '%s'. Using hardcoded defaults.",
                self._path,
            )
            self._cache = {}
        except Exception as exc:
            logger.warning(
                "[PromptManager] Failed to load prompts.yaml: %s. Using hardcoded defaults.",
                exc,
            )
            self._cache = {}

    def _lookup(self, agent_name: str, template_name: str) -> str:
        """Look up a template in the YAML cache or fall back to defaults.

        Args:
            agent_name: Agent section name.
            template_name: Template key within that section.

        Returns:
            Raw (unformatted) template string.
        """
        # Try YAML cache first.
        agent_section = self._cache.get(agent_name) or {}
        value = agent_section.get(template_name)
        if value:
            return str(value).strip()

        # Fall back to hardcoded defaults.
        default_section = _DEFAULTS.get(agent_name) or {}
        return default_section.get(template_name) or ""


class _SafeFormatMap(dict):
    """A :class:`dict` subclass that returns the key placeholder on missing keys.

    Used with :meth:`str.format_map` so that templates with unset placeholders
    are returned unchanged rather than raising :class:`KeyError`.
    """

    def __missing__(self, key: str) -> str:
        """Return the original ``{key}`` placeholder when the key is absent.

        This intentionally prevents :meth:`str.format_map` from raising
        :class:`KeyError` on partially-populated templates so callers receive
        the best-effort formatted string instead of an exception.
        """
        return f"{{{key}}}"


# Module-level singleton for convenience.
_default_manager: Optional[PromptManager] = None


def get_default_prompt_manager() -> PromptManager:
    """Return the module-level :class:`PromptManager` singleton.

    Returns:
        Shared :class:`PromptManager` instance (created on first call).
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager
