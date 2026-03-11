# -*- coding: utf-8 -*-
"""SMILES data validation and resolution utilities.

This module validates that the H5 data file and required SMILES/molecule
columns are available before any pipeline stage attempts to use them.

Public API
----------
- validate_h5_smiles_path(config: dict) -> str
- validate_h5_columns(h5_path: str) -> List[str]
- resolve_smiles(config: dict) -> List[str]
- class ConfigurationError(Exception)
"""

from __future__ import annotations

import os
import string
from typing import Any, Dict, List


# =============================================================================
# Logging Helper
# =============================================================================


def _log(msg: str, *, console: bool = False) -> None:
    """Unified logging output.

    Args:
        msg: Message to log.
        console: If True, adds [CELL_CONSOLE] prefix; otherwise [DETAIL].
    """
    summary_only = str(os.environ.get("CELL_SUMMARY_ONLY", "0")).lower() in {"1", "true", "yes"}
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    elif not summary_only:
        print(f"[DETAIL] {msg}", flush=True)


# =============================================================================
# Custom Exception
# =============================================================================


class ConfigurationError(Exception):
    """Raised when pipeline configuration is invalid or incomplete."""


# =============================================================================
# Constants
# =============================================================================

#: The five datasets that MUST exist inside the HDF5 ``combined`` group.
MANDATORY_HDF5_DATASETS: List[str] = [
    "morphology_pre",
    "morphology_post",
    "smiles",
    "dose",
    "split_id",
]


# =============================================================================
# Internal Helpers
# =============================================================================


def _expand_template(template: str, config: dict) -> str:
    """Substitute ${key} placeholders using top-level config values.

    Args:
        template: String that may contain ``${key}`` placeholders.
        config: Top-level configuration dict supplying substitution values.

    Returns:
        The template with all recognised placeholders replaced.
    """
    # Use string.Template style expansion with ${key} syntax.
    result = template
    for key, value in config.items():
        if isinstance(value, str):
            result = result.replace(f"${{{key}}}", value)
    return result


# =============================================================================
# Public API
# =============================================================================


def validate_h5_smiles_path(config: dict) -> str:
    """Determine and validate the path to the H5 data file.

    Resolution order:
    1. ``paths.data_h5_path`` if set and not null.
    2. Constructed from ``paths.data_root`` + ``paths.data_h5_filename``
       with ``${dataset_name}`` / ``${split_name}`` placeholders expanded.

    Args:
        config: Top-level pipeline configuration dict (from
            ``pipeline_config.json``).

    Returns:
        Absolute or relative path to the H5 file that exists on disk.

    Raises:
        ConfigurationError: If the resolved path does not exist on disk.
    """
    paths: Dict[str, Any] = config.get("paths") or {}

    h5_path: str | None = paths.get("data_h5_path") or None

    if not h5_path:
        # Construct from data_root + data_h5_filename template.
        data_root: str = paths.get("data_root") or "./data"
        filename_template: str = paths.get("data_h5_filename") or ""
        if not filename_template:
            raise ConfigurationError(
                "Cannot resolve H5 path: 'paths.data_h5_filename' is not set "
                "in pipeline_config.json and 'paths.data_h5_path' is null."
            )
        filename = _expand_template(filename_template, config)
        h5_path = os.path.join(data_root, filename)

    _log(f"[SMILES] Resolved H5 path: {h5_path}")

    if not os.path.isfile(h5_path):
        raise ConfigurationError(
            f"H5 data file not found at: {h5_path}. "
            "Check paths.data_h5_path in pipeline_config.json"
        )

    return h5_path


def validate_h5_columns(h5_path: str) -> List[str]:
    """Open an H5 file and verify that all mandatory datasets are present.

    Navigates to the ``combined`` group if it exists, otherwise falls back to
    root-level keys.  Validates that every dataset in
    :data:`MANDATORY_HDF5_DATASETS` is present and raises
    :class:`ConfigurationError` with an actionable message if any are missing.

    Args:
        h5_path: Path to the H5 file to inspect.

    Returns:
        List of all key names found inside the ``combined`` group (or root).

    Raises:
        ConfigurationError: If one or more mandatory datasets are missing.
        ImportError: If the ``h5py`` package is not installed.
    """
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'h5py' package is required to read H5 files. "
            "Install it with: pip install h5py"
        ) from exc

    with h5py.File(h5_path, "r") as f:
        root_keys: List[str] = list(f.keys())
        _log(f"[SMILES] Root-level keys in {h5_path}: {root_keys}")

        if "combined" in f:
            group = f["combined"]
            columns: List[str] = list(group.keys())
            _log(f"[SMILES] Keys inside 'combined' group: {columns}")
        else:
            _log(
                f"[SMILES] No 'combined' group found in {h5_path}; "
                "using root-level keys."
            )
            columns = root_keys

    lower_cols = [c.lower() for c in columns]
    missing = [
        ds for ds in MANDATORY_HDF5_DATASETS if ds.lower() not in lower_cols
    ]
    if missing:
        raise ConfigurationError(
            f"Missing mandatory HDF5 datasets in '{h5_path}' "
            f"(group: 'combined'): {missing}. "
            f"Available keys: {columns}. "
            "Ensure the HDF5 file contains a 'combined' group with all "
            f"required datasets: {MANDATORY_HDF5_DATASETS}"
        )

    return columns


def resolve_smiles(config: dict) -> List[str]:
    """Validate configuration and return the list of SMILES strings.

    Calls :func:`validate_h5_smiles_path` and :func:`validate_h5_columns`
    then reads and returns the ``smiles`` dataset from the ``combined`` group
    (or root level) of the H5 file.

    Args:
        config: Top-level pipeline configuration dict.

    Returns:
        List of SMILES strings read from the H5 file.

    Raises:
        ConfigurationError: If path or column validation fails.
        ImportError: If ``h5py`` or ``numpy`` is not installed.
    """
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'h5py' and 'numpy' packages are required to read H5 files. "
            "Install them with: pip install h5py numpy"
        ) from exc

    h5_path = validate_h5_smiles_path(config)
    # validate_h5_columns also confirms all mandatory datasets exist.
    validate_h5_columns(h5_path)

    with h5py.File(h5_path, "r") as f:
        # Prefer the 'combined' group; fall back to root.
        if "combined" in f:
            group = f["combined"]
        else:
            group = f

        # Read the 'smiles' dataset (validated to exist by validate_h5_columns).
        lower_keys = {k.lower(): k for k in group.keys()}
        col_name = lower_keys.get("smiles", "smiles")
        raw = group[col_name][:]

    # Decode bytes to str if necessary.
    smiles_list: List[str] = []
    for item in np.asarray(raw).flatten():
        if isinstance(item, (bytes, np.bytes_)):
            smiles_list.append(item.decode("utf-8", errors="replace"))
        else:
            smiles_list.append(str(item))

    _log(f"[SMILES] Loaded {len(smiles_list)} SMILES strings from '{col_name}' dataset.")
    return smiles_list
