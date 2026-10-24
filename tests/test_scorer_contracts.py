"""Exercise every shipped scorer's own runnable example, without model downloads."""
import pytest

from benchmark_scoring.registry import SCORERS, catalog, fingerprint, score_task_detail, validate_task
from scripts.run_remote_benchmark import _validate_benchmark_tasks


@pytest.mark.parametrize("scorer", [s for s in SCORERS.values() if s.graded], ids=lambda s: s.id)
def test_registered_scorer_accepts_reference_and_rejects_wrong_answer(scorer):
    task = {**scorer.example, "task_type": scorer.id}
    validate_task(task)
    assert score_task_detail(task["response"], task).score == 1
    assert score_task_detail("", task).score == 0
    assert score_task_detail("unrelated incorrect response", task).score == 0


@pytest.mark.parametrize("tasks", [
    [], [{}], [{"id": "x", "prompt": ""}],
    [{"id": "x", "prompt": "Answer.", "task_type": "unknown"}],
    [{"id": "x", "prompt": "Answer.", "task_type": "numeric_exact", "expected_answer": float("nan")}],
    [{"id": "x", "prompt": "Answer.", "task_type": "numeric_exact", "expected_answer": 4}] * 2,
])
def test_invalid_tasks_rejected_before_engine_execution(tasks):
    with pytest.raises(ValueError):
        _validate_benchmark_tasks(tasks)


def test_catalog_and_fingerprint_are_stable_and_callable():
    assert fingerprint() == fingerprint()
    selectable = {entry["id"] for entry in catalog()}
    assert selectable == {entry.id for entry in SCORERS.values() if entry.graded}
    assert all(callable(SCORERS[name].evaluate) for name in selectable)
