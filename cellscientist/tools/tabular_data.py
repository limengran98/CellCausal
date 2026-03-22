from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_TABULAR_SUFFIXES = (".csv", ".tsv", ".parquet", ".xlsx", ".xls")

_PATH_RE = re.compile(
    r"(?P<path>(?:\.{1,2}/|/)?[^\s\"':,;]+?\.(?:csv|tsv|parquet|xlsx|xls))",
    re.IGNORECASE,
)

_GENERIC_DATA_KEYWORDS = (
    "csv",
    "tsv",
    "parquet",
    "excel",
    "spreadsheet",
    "table",
    "dataframe",
    "my own data",
    "my data",
    "user data",
    "tabular data",
    "读取这个数据表",
    "读取数据",
    "自己的数据",
    "数据表",
    "表格",
    "探索分析",
    "做验证",
)


def extract_tabular_path_from_text(text: str) -> str | None:
    """Extract a likely user-provided tabular file path from free text."""

    matches = _PATH_RE.findall(text or "")
    if not matches:
        return None
    return matches[0]


def resolve_tabular_path(path: str | None, *, base_dir: str | os.PathLike[str] | None = None) -> str | None:
    """Resolve a user path against the repo root and current working directory."""

    if not path:
        return None

    raw = Path(path)
    if raw.is_absolute():
        return str(raw)

    candidates = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / raw)
    candidates.append(Path.cwd() / raw)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((candidates[0] if candidates else raw).resolve())


def looks_like_generic_data_reference(text: str) -> bool:
    lowered = (text or "").lower()
    if extract_tabular_path_from_text(text):
        return True
    return any(keyword in lowered for keyword in _GENERIC_DATA_KEYWORDS)


def is_supported_tabular_path(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in SUPPORTED_TABULAR_SUFFIXES


def profile_tabular_file(path: str) -> dict[str, Any]:
    """Read and minimally profile a user-provided table file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return {
            "status": "missing_input_path",
            "input_path": str(resolved),
            "file_type": resolved.suffix.lower().lstrip(".") or "unknown",
            "shape": None,
            "columns": [],
            "column_types": {},
            "missingness_summary": {},
            "suggested_analysis_modes": [],
            "message": "The provided data file could not be found.",
        }

    file_type = resolved.suffix.lower().lstrip(".")
    if f".{file_type}" not in SUPPORTED_TABULAR_SUFFIXES:
        return {
            "status": "unsupported_file_type",
            "input_path": str(resolved),
            "file_type": file_type or "unknown",
            "shape": None,
            "columns": [],
            "column_types": {},
            "missingness_summary": {},
            "suggested_analysis_modes": [],
            "message": "This file type is not supported by the minimal generic data intake path.",
        }

    try:
        if file_type == "csv":
            frame = pd.read_csv(resolved)
        elif file_type == "tsv":
            frame = pd.read_csv(resolved, sep="\t")
        elif file_type == "parquet":
            frame = pd.read_parquet(resolved)
        elif file_type in {"xlsx", "xls"}:
            frame = pd.read_excel(resolved)
        else:
            raise ValueError(f"Unsupported tabular format: {file_type}")
    except ImportError as exc:
        return {
            "status": "unsupported_runtime_dependency",
            "input_path": str(resolved),
            "file_type": file_type,
            "shape": None,
            "columns": [],
            "column_types": {},
            "missingness_summary": {},
            "suggested_analysis_modes": [],
            "message": "The file type is recognized, but the local runtime is missing the required reader dependency.",
            "details": {"error": str(exc)},
        }
    except Exception as exc:
        return {
            "status": "read_failed",
            "input_path": str(resolved),
            "file_type": file_type,
            "shape": None,
            "columns": [],
            "column_types": {},
            "missingness_summary": {},
            "suggested_analysis_modes": [],
            "message": "The data file was found, but profiling failed during read-in.",
            "details": {"error": str(exc)},
        }

    column_types = {column: str(dtype) for column, dtype in frame.dtypes.items()}
    missing_by_column = {column: int(value) for column, value in frame.isna().sum().to_dict().items()}
    missing_columns = {column: count for column, count in missing_by_column.items() if count > 0}

    return {
        "status": "profiled",
        "input_path": str(resolved),
        "file_type": file_type,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(column) for column in frame.columns.tolist()],
        "column_types": column_types,
        "missingness_summary": {
            "total_missing_cells": int(frame.isna().sum().sum()),
            "columns_with_missing": missing_columns,
            "fully_observed_columns": [
                column for column, count in missing_by_column.items() if count == 0
            ],
        },
        "suggested_analysis_modes": suggest_analysis_modes(frame),
    }


def suggest_analysis_modes(frame: pd.DataFrame) -> list[str]:
    """Derive a few rule-based analysis modes from a profiled dataframe."""

    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    datetime_columns = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical_columns = [
        column
        for column in frame.columns
        if column not in numeric_columns and column not in datetime_columns
    ]

    target_candidates = [
        column
        for column in frame.columns
        if str(column).lower() in {"target", "label", "class", "outcome", "response", "y"}
    ]

    modes = ["tabular_summary"]
    if numeric_columns:
        modes.append("numeric_eda")
    if categorical_columns:
        modes.append("categorical_breakdown")
    if datetime_columns:
        modes.append("time_series_screen")

    if target_candidates:
        target_column = target_candidates[0]
        unique_count = int(frame[target_column].nunique(dropna=True))
        if target_column in numeric_columns and unique_count > 10:
            modes.append("baseline_regression")
        else:
            modes.append("baseline_classification")
    elif numeric_columns and categorical_columns:
        modes.append("grouped_comparison")

    return modes
