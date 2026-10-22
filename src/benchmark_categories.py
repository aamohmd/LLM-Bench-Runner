"""Domain-neutral benchmark execution defaults.

Suites, categories, task requirements, and quality thresholds are registry data
owned by the control plane. The execution package intentionally contains no
static workload catalog.
"""

from __future__ import annotations


MINIMUM_REPETITIONS = 3


def get_system_prompt_for_task(task: dict, prompt_strategy: str | None = None) -> str:
    """Use task-supplied text; registered prompt strategies are composed by the API."""
    return str(task.get("system") or "")


def validate_self_contained_tasks(tasks: dict | list[dict]) -> None:
    """Reject external retrieval requests before executing a benchmark."""
    for task in tasks.values() if isinstance(tasks, dict) else tasks:
        if (
            task.get("use_rag") is True
            or task.get("retrieval_query") is not None
            or task.get("retrieval_corpus") is not None
        ):
            raise ValueError(
                "External retrieval is not supported. Include the required source "
                "context directly in the registered task prompt."
            )


def validate_approval_request(*, repetitions: int, **_: object) -> None:
    """Validate only execution invariants; suite coverage comes from registry data."""
    if repetitions < MINIMUM_REPETITIONS:
        raise ValueError(
            f"Approval runs require at least {MINIMUM_REPETITIONS} repetitions "
            f"(got {repetitions})."
        )


def concurrency_scope(concurrency: int, required_levels: tuple[int, ...] | list[int] | None = None) -> str:
    """Classify concurrency using levels supplied by the selected suite or run plan."""
    if required_levels is None:
        return "required"
    return "required" if concurrency in required_levels else "exploratory"
