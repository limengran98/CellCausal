from types import SimpleNamespace

from cellscientist.core.cell_naming import infer_task_name_from_cell


def _code_cell(src: str, subtask_name=None):
    md = {}
    if subtask_name is not None:
        md["subtask"] = {"name": subtask_name}
    return SimpleNamespace(source=src, metadata=md)


def test_infer_task_name_prefers_explicit_subtask_name():
    c = _code_cell("print('x')", subtask_name="Custom Task")
    assert infer_task_name_from_cell(c, 0) == "Custom Task"


def test_infer_task_name_from_leading_comment():
    c = _code_cell("# Train EquiMorph model\nprint('run')")
    assert infer_task_name_from_cell(c, 1) == "Train EquiMorph model"


def test_infer_task_name_heuristics_training_loop():
    c = _code_cell("for ep in range(10):\n    model.train()\n    loss.backward()")
    assert infer_task_name_from_cell(c, 2) == "Training Loop"


def test_infer_task_name_default_fallback():
    c = _code_cell("x = 1")
    assert infer_task_name_from_cell(c, 5) == "Cell_6"
