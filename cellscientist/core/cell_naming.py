from __future__ import annotations

from typing import Any


def infer_task_name_from_cell(cell: Any, cell_idx: int) -> str:
    """Infer a readable task name when notebook subtask metadata is missing."""
    try:
        metadata = getattr(cell, "metadata", {}) or {}
        task_meta = metadata.get("subtask", {}) if hasattr(metadata, "get") else {}
        explicit = (task_meta.get("name") if isinstance(task_meta, dict) else None) or ""
        if str(explicit).strip():
            return str(explicit).strip()
    except Exception:
        pass

    src = str(getattr(cell, "source", "") or "").strip()
    if not src:
        return f"Cell_{cell_idx + 1}"

    first_line = src.splitlines()[0].strip()
    if first_line.startswith("#") and len(first_line) > 1:
        title = first_line.lstrip("#").strip()
        if title:
            return title[:64]

    low = src.lower()

    def has(*tokens: str) -> bool:
        return any(t in low for t in tokens)

    if has("import ", "from ") and len(src.splitlines()) <= 40:
        return "Setup & Imports"
    if has("h5py.file(", "load_data", "stage1_h5_path", "dataset(", "dataloader"):
        return "Data Loading"
    if has("class ", "nn.module", "def forward", "multiheadattention"):
        return "Model Definition"
    if has("model.train(", "loss.backward", "optim.", "optimizer", "for ep in range"):
        return "Training Loop"
    if has("model.eval(", "pearsonr", "r2_score", "calc_metrics", "mean_squared_error"):
        return "Evaluation"
    if has("json.dump", "metrics.json", "analysis_summary.json", "experiment_report.md"):
        return "Export Artifacts"

    return f"Cell_{cell_idx + 1}"
