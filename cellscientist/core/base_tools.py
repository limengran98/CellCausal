# -*- coding: utf-8 -*-
"""Shared utility functions used across multiple pipeline modules (DRY).

Extracts repeated patterns from ``bio_kb/smiles_resolver.py``,
``llm_client.py``, and elsewhere into a single importable location.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# H5 data loading
# =============================================================================


def load_h5_data(h5_path: Path) -> Dict[str, Any]:
    """Load an HDF5 file and return a dict mapping dataset names to arrays.

    Requires ``h5py``; returns an empty dict and logs a warning if the
    library is unavailable or the file cannot be read.

    Args:
        h5_path: Path to the ``.h5`` or ``.hdf5`` file.

    Returns:
        Dict mapping dataset/group names to their contents (numpy arrays or
        nested dicts for groups).
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        logger.warning("[base_tools] h5py is not installed; returning empty dict.")
        return {}

    result: Dict[str, Any] = {}
    try:
        with h5py.File(Path(h5_path), "r") as f:
            def _walk(group: Any, target: Dict[str, Any]) -> None:
                for key in group.keys():
                    item = group[key]
                    if hasattr(item, "keys"):
                        # It's a group — recurse.
                        sub: Dict[str, Any] = {}
                        _walk(item, sub)
                        target[key] = sub
                    else:
                        target[key] = item[()]

            _walk(f, result)
    except Exception as exc:
        logger.warning("[base_tools] Failed to load H5 file '%s': %s", h5_path, exc)
    return result


# =============================================================================
# SMILES extraction from H5
# =============================================================================


def extract_smiles_from_h5(h5_path: Path) -> List[str]:
    """Extract SMILES strings from an HDF5 file.

    Delegates to :func:`.smiles_resolver.validate_h5_columns`
    (``cellscientist.core.smiles_resolver``) and the column-discovery logic
    in the core smiles resolver.

    Args:
        h5_path: Path to the ``.h5`` file.

    Returns:
        List of SMILES strings (may be empty on failure).
    """
    try:
        from .smiles_resolver import validate_h5_columns  # type: ignore

        columns = validate_h5_columns(str(h5_path))
        # Load the raw data and pick the first recognised SMILES column.
        data = load_h5_data(h5_path)
        for col in columns:
            raw = data.get(col)
            if raw is None:
                continue
            # Handle numpy arrays / lists of bytes or strings.
            try:
                items = list(raw)
                smiles: List[str] = []
                for item in items:
                    if isinstance(item, (bytes, bytearray)):
                        smiles.append(item.decode("utf-8", errors="replace"))
                    else:
                        smiles.append(str(item))
                if smiles:
                    return smiles
            except Exception as exc:
                logger.debug("[base_tools] Column '%s' extraction failed: %s", col, exc)
    except Exception as exc:
        logger.warning("[base_tools] extract_smiles_from_h5 failed: %s", exc)
    return []


# =============================================================================
# JSON parsing
# =============================================================================


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from *text*, using the LLM client helper as a fallback.

    Wraps :func:`~cellscientist.core.llm_client.extract_json_from_text` so
    callers don't need to import from ``llm_client`` directly.

    Args:
        text: A string that may or may not be valid JSON (may contain
            markdown fences or extra prose around the JSON object).

    Returns:
        Parsed ``dict`` if successful, ``None`` otherwise.
    """
    # Fast path: direct JSON parse.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Slow path: use the LLM client's extraction helper.
    try:
        from .llm_client import extract_json_from_text  # type: ignore

        return extract_json_from_text(text)
    except Exception as exc:
        logger.debug("[base_tools] safe_json_parse fallback failed: %s", exc)
    return None


# =============================================================================
# Config path resolution
# =============================================================================


def resolve_config_path(config: Dict[str, Any], key_path: List[str]) -> Optional[str]:
    """Navigate a nested config dict safely and return the final string value.

    Replaces repeated ``cfg.get("a", {}).get("b", {}).get("c")`` patterns.

    Args:
        config: Top-level configuration dict.
        key_path: Ordered list of keys to traverse, e.g.
            ``["paths", "data_h5_path"]``.

    Returns:
        The string value at the end of the path, or ``None`` if any
        intermediate key is missing or the final value is falsy.

    Example::

        resolve_config_path(cfg, ["exec", "timeout_seconds"])
        # Returns "300" (or None if not set)
    """
    node: Any = config
    for key in key_path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return str(node) if node is not None else None
