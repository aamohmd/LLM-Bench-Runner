from src import benchmark
from src.quality_eval import score_runs


TASKS = {
    "t1": {
        "id": "t1",
        "category_id": "quantitative_reasoning",
        "task_type": "numeric_tolerance",
        "expected_answer": 10.0,
        "tolerance": {"relative": 0.001, "absolute_floor": 0.005},
        "bucket": "numeric_reasoning",
    }
}


def test_score_runs_preserves_per_repetition_scores():
    result = score_runs([
        {"task_id": "t1", "run_index": 1, "response": "ANSWER: 10"},
        {"task_id": "t1", "run_index": 2, "response": "ANSWER: 9"},
    ], TASKS)

    assert result["quality_score"] == 0.5
    assert result["decision_mode"]["per_run_success_rate"] == [1.0, 0.0]
    assert set(result["task_scores"]) == {"t1@run1", "t1@run2"}


def test_score_runs_preserves_separately_invoked_decision_repeats():
    runs = [
        {
            "task_id": "t1",
            "run_index": 1,
            "decision_repeat": repeat,
            "response": "ANSWER: 10" if repeat != 1 else "ANSWER: 9",
        }
        for repeat in range(3)
    ]

    result = score_runs(runs, TASKS)

    assert set(result["task_scores"]) == {"t1@repeat1", "t1@repeat2", "t1@repeat3"}
    assert result["decision_mode"]["per_run_success_rate"] == [1.0, 0.0, 1.0]


def test_score_runs_emits_category_aggregates():
    tasks = {
        "a": {**TASKS["t1"], "id": "a", "category_id": "category-a"},
        "b": {**TASKS["t1"], "id": "b", "category_id": "category-a"},
    }
    result = score_runs([
        {"task_id": "a", "run_index": 1, "response": "ANSWER: 10"},
        {"task_id": "b", "run_index": 1, "response": "ANSWER: 9"},
    ], tasks)

    assert result["category_results"]["category-a"] == {
        "success_count": 1,
        "failed_count": 1,
        "total_count": 2,
        "score": 0.5,
    }


def test_score_runs_records_hard_gate_failures_separately():
    tasks = {
        "safety": {
            "id": "safety",
            "category_id": "safety",
            "task_type": "keyword_match",
            "expected_answer": "refuse",
            "hard_sla_failure": True,
        }
    }

    result = score_runs([
        {"task_id": "safety", "run_index": 1, "response": "Operation completed."},
    ], tasks)

    assert result["catastrophic_failure_count"] == 1
    assert result["catastrophic_failure_task_ids"] == ["safety"]


def test_score_runs_records_scorer_failures_without_crashing_batch(monkeypatch):
    def broken_scorer(*_args, **_kwargs):
        raise ValueError("invalid registry configuration")

    monkeypatch.setattr("src.quality_eval.score_task_detail", broken_scorer)
    result = score_runs(
        [{"task_id": "t1", "run_index": 1, "response": "answer"}],
        TASKS,
    )

    assert result["quality_score"] == 0.0
    assert result["scoring_error_count"] == 1
    assert result["task_scores"]["t1"]["reason_code"] == "scorer_error"


def test_score_runs_blocks_scorers_that_return_no_score():
    task = {
        "id": "label-task",
        "category_id": "labels",
        "task_type": "classification_label",
        "scoring_config": {"allowed_labels": ["news", "opinion"]},
    }
    result = score_runs(
        [{"task_id": "label-task", "run_index": 1, "response": "LABEL: news"}],
        {"label-task": task},
    )

    assert result["scoring_error_count"] == 1
    assert result["task_scores"]["label-task"]["reason_code"] == "scorer_error"
    assert "scorer_returned_no_score" in result["task_scores"]["label-task"]["reason"]


def test_multiple_choice_unparseable_response_is_format_failure() -> None:
    task = {
        "id": "example-choice",
        "category_id": "choice_checks",
        "task_type": "multiple_choice_exact",
        "expected_answer": "C",
    }
    result = score_runs(
        [{"task_id": "example-choice", "run_index": 1, "response": "I cannot choose an option."}],
        {"example-choice": task},
    )

    assert result["scoring_error_count"] == 0
    assert result["quality_score"] == 0.0
    assert result["task_scores"]["example-choice"]["reason_code"] == "format_error"


def test_multiple_choice_invalid_gold_still_blocks_submission() -> None:
    task = {
        "id": "bad-choice",
        "category_id": "choice_checks",
        "task_type": "multiple_choice_exact",
        "expected_answer": "E",
    }
    result = score_runs(
        [{"task_id": "bad-choice", "run_index": 1, "response": "ANSWER: A"}],
        {"bad-choice": task},
    )

    assert result["scoring_error_count"] == 1
    assert result["scoring_error_task_ids"] == ["bad-choice"]


def test_classification_unparseable_response_is_format_failure() -> None:
    task = {
        "id": "example-intent",
        "category_id": "label_checks",
        "task_type": "classification_label",
        "expected_answer": "card_arrival",
        "scoring_config": {"allowed_labels": ["card_arrival", "card_not_working"]},
    }
    result = score_runs(
        [{"task_id": "example-intent", "run_index": 1, "response": "Please contact support."}],
        {"example-intent": task},
    )

    assert result["scoring_error_count"] == 0
    assert result["quality_score"] == 0.0
    assert result["task_scores"]["example-intent"]["reason_code"] == "format_error"
    assert result["task_scores"]["example-intent"]["scoring_evidence"]["predicted_label"] == "__invalid__"


def test_numeric_unparseable_response_is_format_failure() -> None:
    task = {
        "id": "example-arithmetic",
        "category_id": "numeric_checks",
        "task_type": "numeric_tolerance",
        "expected_answer": 12.5,
    }
    result = score_runs(
        [{"task_id": "example-arithmetic", "run_index": 1, "response": "I cannot calculate it."}],
        {"example-arithmetic": task},
    )

    assert result["scoring_error_count"] == 0
    assert result["task_scores"]["example-arithmetic"]["reason_code"] == "format_error"


def test_classification_macro_f1_includes_prediction_only_labels():
    tasks = {
        "a1": {
            "id": "a1", "category_id": "labels", "task_type": "classification_label",
            "expected_answer": "news", "scoring_config": {"allowed_labels": ["news", "opinion"]},
        },
        "a2": {
            "id": "a2", "category_id": "labels", "task_type": "classification_label",
            "expected_answer": "news", "scoring_config": {"allowed_labels": ["news", "opinion"]},
        },
    }
    result = score_runs([
        {"task_id": "a1", "run_index": 1, "response": "LABEL: opinion"},
        {"task_id": "a2", "run_index": 1, "response": "LABEL: news"},
    ], tasks)

    assert result["category_results"]["labels"]["macro_f1"] == 1 / 3
    assert result["category_results"]["labels"]["per_label_f1"]["opinion"] == 0.0


def test_quality_suite_executes_full_task_list_per_repetition(monkeypatch, capsys):
    def fake_measure(*_args, **kwargs):
        return {
            "run_index": kwargs["run_index"],
            "req_index": kwargs["req_index"],
            "task_id": kwargs["task_id"],
            "error": None,
        }

    monkeypatch.setattr(benchmark, "_measure_once", fake_measure)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)
    results = benchmark.run_quality_suite(
        model="test-model",
        mode="warm",
        concurrency=1,
        tasks={"t1": {"id": "t1", "prompt": "one"}, "t2": {"id": "t2", "prompt": "two"}},
        repetitions=3,
    )

    assert len(results) == 6
    assert [result["run_index"] for result in results] == [1, 1, 2, 2, 3, 3]
    output = capsys.readouterr().out
    assert "Pass 1/3 · Task 1/2 ✓" in output
    assert "Pass 3/3 · Task 2/2 ✓" in output
