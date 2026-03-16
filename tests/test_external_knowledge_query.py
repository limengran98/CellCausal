from cellscientist.core.external_knowledge_mirothink import _build_query


def _cfg():
    return {
        "literature": {
            "task_keywords": "cell painting perturbation modeling",
            "query_term_limit": 12,
        }
    }


def test_build_query_sanitizes_code_scaffold_from_hint():
    q = _build_query(
        context_text="",
        stage="review",
        cfg=_cfg(),
        query_hint="CELL INDEX READ-ONLY CONTEXT import sys h5py numpy pandas torch from utils review feedback",
    )
    ql = q.lower()
    assert "cell index" not in ql
    assert "import sys" not in ql
    assert "h5py" not in ql
    assert "review" in ql
    assert "feedback" in ql


def test_build_query_uses_safe_default_when_hint_is_noise_only():
    q = _build_query(
        context_text="",
        stage="review",
        cfg=_cfg(),
        query_hint="CELL INDEX READ ONLY CONTEXT import sys os numpy torch",
    )
    assert "single-cell perturbation response modeling" in q.lower()
