"""Domain-neutral quality scoring dispatcher."""

from __future__ import annotations

import json
import re
import statistics
from typing import Iterable


from benchmark_scoring.registry import SCORERS, score_task_detail
from benchmark_scoring.scorers import ScoreDetail


def score_task(response: str, task: dict) -> tuple[float | None, str]:
    return score_task_detail(response, task).as_tuple()


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def score_runs(
    runs: Iterable[dict],
    tasks: dict[str, dict],
    capture_traces: bool = False,
) -> dict:
    """Score a set of completed benchmark runs against their tasks."""
    runs = list(runs)
    task_scores: dict[str, dict] = {}
    bucket_metrics: dict[str, dict] = {}
    repetition_metrics: dict[int, dict[str, int]] = {}
    task_occurrences: dict[str, int] = {}
    score_key_occurrences: dict[str, int] = {}
    for run in runs:
        task_id = run.get("task_id")
        if task_id in tasks:
            task_occurrences[task_id] = task_occurrences.get(task_id, 0) + 1
    success_count = 0
    failed_count = 0
    total_count = 0
    category_metrics: dict[str, dict[str, int | float | None]] = {}
    classification_pairs: dict[str, list[tuple[str, str]]] = {}
    multilabel_counts: dict[str, dict[str, int]] = {}
    multilabel_records: dict[str, list[tuple[set[str], set[str], set[str]]]] = {}
    span_counts: dict[str, dict[str, int]] = {}
    edit_counts: dict[str, dict[str, int]] = {}
    catastrophic_failure_task_ids: list[str] = []
    scoring_error_task_ids: list[str] = []
    ungraded_count = 0
    non_accuracy_categories: dict[str, set[str]] = {}
    non_accuracy_buckets: set[str] = set()
    rating_maes: dict[str, list[float]] = {}
    category_quality_totals: dict[str, float] = {}
    category_task_types: dict[str, set[str]] = {}

    def rubric_fields(task: dict) -> dict:
        """Persist only the task metadata needed for approval-rubric aggregation."""
        program = task.get("program")
        return {
            "category_id": task.get("category_id", task.get("task_category", "unknown")),
            "language": task.get("language"),
            "task_type": task.get("task_type"),
            "bucket": task.get("bucket", "unknown"),
            "program_chain_depth": len(set(re.findall(r"#\d+", program))) + 1 if isinstance(program, str) else None,
            "trap": bool(task.get("trap_explanation")),
            "difficulty_tag": task.get("difficulty_tag"),
            "attack_family": task.get("attack_family"),
            "severity": task.get("severity"),
            "readiness_status": task.get("readiness_status", "implementation_ready"),
            "hard_sla_failure": task.get("hard_sla_failure", False),
            "has_distractor": bool(task.get("distractor_citations") or task.get("distractor_source_ids")),
            "has_faithfulness_evidence": bool(task.get("faithfulness_expected")),
        }

    def collect_metric_evidence(category_id: str, task: dict, detail: ScoreDetail) -> None:
        gold_label = detail.evidence.get("gold_label")
        predicted_label = detail.evidence.get("predicted_label")
        if isinstance(gold_label, str) and isinstance(predicted_label, str):
            classification_pairs.setdefault(category_id, []).append((gold_label, predicted_label))

        count_target = None
        if task.get("task_type") == "multilabel_f1":
            count_target = multilabel_counts.setdefault(category_id, {"tp": 0, "fp": 0, "fn": 0})
            expected_labels = detail.evidence.get("expected_labels")
            predicted_labels = detail.evidence.get("predicted_labels")
            allowed_labels = detail.evidence.get("allowed_labels")
            if all(isinstance(values, list) for values in (expected_labels, predicted_labels, allowed_labels)):
                multilabel_records.setdefault(category_id, []).append(
                    (set(expected_labels), set(predicted_labels), set(allowed_labels))  # type: ignore[arg-type]
                )
        elif task.get("task_type") == "span_labels":
            count_target = span_counts.setdefault(category_id, {"tp": 0, "fp": 0, "fn": 0})
        elif task.get("task_type") == "copy_edit_f05":
            count_target = edit_counts.setdefault(category_id, {"tp": 0, "fp": 0, "fn": 0})
        if count_target is not None:
            count_target["tp"] += int(detail.evidence.get("true_positive", 0))
            count_target["fp"] += int(detail.evidence.get("false_positive", 0))
            count_target["fn"] += int(detail.evidence.get("false_negative", 0))

    def raw_trace(task: dict, actual: object) -> dict[str, str]:
        expected = task.get("expected_answer", "")
        return {
            "prompt": str(task.get("prompt") or task.get("user") or ""),
            "expected": (
                expected
                if isinstance(expected, str)
                else json.dumps(expected, ensure_ascii=False, sort_keys=True)
            ),
            "actual": str(actual or ""),
        }

    for run in runs:
        task_id = run.get("task_id")
        if task_id not in tasks:
            continue

        task = tasks[task_id]
        category_id = task.get("category_id", task.get("task_category", "unknown"))
        category_task_types.setdefault(category_id, set()).add(task.get("task_type", ""))
        category_metric = category_metrics.setdefault(
            category_id,
            {"success_count": 0, "failed_count": 0, "total_count": 0, "score": None},
        )
        total_count += 1
        category_metric["total_count"] = int(category_metric["total_count"]) + 1
        if task.get("task_type") in {"editorial_review", "editorial_ratings"}:
            non_accuracy_categories.setdefault(category_id, set()).add(
                "ungraded_generation" if task["task_type"] == "editorial_review" else "expert_rating_agreement"
            )
        run_index = int(run.get("run_index", 1))
        decision_repeat = run.get("decision_repeat")
        repetition_id = int(decision_repeat) + 1 if decision_repeat is not None else run_index
        repetition = repetition_metrics.setdefault(repetition_id, {"success_count": 0, "total_count": 0})
        repetition["total_count"] += 1
        if task_occurrences[task_id] == 1:
            score_key_base = task_id
        elif decision_repeat is not None:
            score_key_base = f"{task_id}@repeat{repetition_id}"
        else:
            score_key_base = f"{task_id}@run{run_index}"
        score_key_occurrences[score_key_base] = score_key_occurrences.get(score_key_base, 0) + 1
        score_key = score_key_base
        if score_key_occurrences[score_key_base] > 1:
            score_key = f"{score_key_base}#{score_key_occurrences[score_key_base]}"
        bucket = task.get("bucket", "unknown")
        if task.get("task_type") in {"editorial_review", "editorial_ratings"}:
            non_accuracy_buckets.add(bucket)
        if bucket not in bucket_metrics:
            bucket_metrics[bucket] = {"success_count": 0, "failed_count": 0, "total_count": 0}
        bucket_metrics[bucket]["total_count"] += 1

        # Infrastructure/engine failure
        if run.get("error") or "response" not in run:
            category_metric["execution_errors"] = int(category_metric.get("execution_errors", 0)) + 1
            if task.get("task_type") in {
                "classification_label", "multilabel_f1", "span_labels", "copy_edit_f05"
            }:
                try:
                    collect_metric_evidence(category_id, task, score_task_detail("", task))
                except Exception:
                    scoring_error_task_ids.append(score_key)
            failed_count += 1
            category_metric["failed_count"] = int(category_metric["failed_count"]) + 1
            bucket_metrics[bucket]["failed_count"] += 1
            task_score = {
                "score": None,
                "reason": run.get("error") or "no_response_produced",
                "task_category": category_id,
                "bucket": bucket,
                "run_index": run_index,
                "failed": True,
                "extraction_path": None,
                "rubric": rubric_fields(task),
                "reason_code": "execution_error",
                "scoring_evidence": {},
            }
            if capture_traces:
                task_score["raw_trace"] = raw_trace(
                    task, run.get("response") or run.get("error")
                )
            task_scores[score_key] = task_score
            continue

        # Score the response
        try:
            detail = score_task_detail(run["response"], task)
        except Exception as exc:
            detail = ScoreDetail(
                None,
                f"scorer_error:{type(exc).__name__}",
                "scorer_error",
                {"error_type": type(exc).__name__, "error_message": str(exc)},
            )
        intentional_ungraded = detail.reason_code == "ungraded_generation"
        if detail.score is None and not intentional_ungraded:
            if detail.reason_code != "scorer_error":
                detail = ScoreDetail(
                    None,
                    f"scorer_returned_no_score:{detail.reason}",
                    "scorer_error",
                    {**detail.evidence, "error_type": "MissingScore"},
                )
            scoring_error_task_ids.append(score_key)
        score, reason = detail.score, detail.reason
        collect_metric_evidence(category_id, task, detail)
        contribution = float(detail.evidence.get("normalized_quality_score", float(score == 1.0)))
        category_quality_totals[category_id] = category_quality_totals.get(category_id, 0.0) + contribution

        if intentional_ungraded:
            ungraded_count += 1
            category_metric["ungraded_count"] = int(category_metric.get("ungraded_count", 0)) + 1
            category_metric["length_compliant_count"] = int(category_metric.get("length_compliant_count", 0)) + int(detail.evidence["length_compliant"])
        elif score == 1.0:
            success_count += 1
            category_metric["success_count"] = int(category_metric["success_count"]) + 1
            repetition["success_count"] += 1
            bucket_metrics[bucket]["success_count"] += 1
        else:
            failed_count += 1
            category_metric["failed_count"] = int(category_metric["failed_count"]) + 1
            bucket_metrics[bucket]["failed_count"] += 1
            if score == 0.0 and task.get("hard_sla_failure") is True:
                catastrophic_failure_task_ids.append(score_key)

        if task.get("task_type") == "editorial_ratings":
            if "rating_mae" in detail.evidence:
                rating_maes.setdefault(category_id, []).append(float(detail.evidence["rating_mae"]))
            elif detail.reason_code == "format_error":
                category_metric["format_errors"] = int(category_metric.get("format_errors", 0)) + 1

        task_score = {
            "score": score,
            "reason": reason,
            "task_category": category_id,
            "bucket": bucket,
            "run_index": run_index,
            "failed": not intentional_ungraded and score != 1.0,
            "extraction_path": None,
            "rubric": rubric_fields(task),
            "reason_code": detail.reason_code,
            "scoring_evidence": detail.evidence,
        }
        if (capture_traces and score != 1.0) or intentional_ungraded:
            task_score["raw_trace"] = raw_trace(task, run["response"])
        task_scores[score_key] = task_score

    success_rate = (success_count / total_count) if total_count > 0 else 0.0

    # Bucket rates
    for bucket, b_data in bucket_metrics.items():
        b_total = b_data["total_count"]
        b_data["success_rate"] = None if bucket in non_accuracy_buckets else (
            (b_data["success_count"] / b_total) if b_total > 0 else 0.0
        )

    per_run_success_rate = [
        repetition["success_count"] / repetition["total_count"]
        for _, repetition in sorted(repetition_metrics.items())
        if repetition["total_count"] > 0
    ]

    for category_metric in category_metrics.values():
        category_total = int(category_metric["total_count"])
        category_metric["score"] = (
            category_quality_totals.get(category_id, 0.0) / category_total if category_total else None
        )

    for category_id, scopes in non_accuracy_categories.items():
        metric = category_metrics[category_id]
        scope = next(iter(scopes)) if len(scopes) == 1 else "mixed_diagnostic"
        if scope == "expert_rating_agreement" and len(category_task_types[category_id]) > 1:
            scope = "normalized_quality"
        metric.update(metric_scope=scope)
        if "ungraded_generation" in scopes:
            metric["score"] = None
        if "expert_rating_agreement" in scopes:
            maes = rating_maes.get(category_id, [])
            metric["mean_rating_mae"] = statistics.mean(maes) if maes else None
            metric["rating_scored_count"] = len(maes)

    for category_id, pairs in classification_pairs.items():
        labels = sorted({label for pair in pairs for label in pair if label != "__invalid__"})
        per_label_f1: dict[str, float] = {}
        for label in labels:
            true_positive = sum(gold == label and predicted == label for gold, predicted in pairs)
            false_positive = sum(gold != label and predicted == label for gold, predicted in pairs)
            false_negative = sum(gold == label and predicted != label for gold, predicted in pairs)
            denominator = 2 * true_positive + false_positive + false_negative
            per_label_f1[label] = (2 * true_positive / denominator) if denominator else 0.0
        category_metrics[category_id]["macro_f1"] = (
            sum(per_label_f1.values()) / len(per_label_f1) if per_label_f1 else None
        )
        category_metrics[category_id]["per_label_f1"] = per_label_f1  # type: ignore[assignment]

    def precision_recall_f1(counts: dict[str, int], beta_sq: float = 1.0) -> tuple[float, float, float]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else float(fn == 0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        score = (
            (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
            if precision or recall else 0.0
        )
        return precision, recall, score

    for category_id, counts in multilabel_counts.items():
        precision, recall, f1 = precision_recall_f1(counts)
        category_metrics[category_id]["multilabel_micro_precision"] = precision
        category_metrics[category_id]["multilabel_micro_recall"] = recall
        category_metrics[category_id]["multilabel_micro_f1"] = f1
        records = multilabel_records.get(category_id, [])
        labels = sorted(set().union(*(allowed for _, _, allowed in records))) if records else []
        per_label_f1: dict[str, float] = {}
        for label in labels:
            label_counts = {
                "tp": sum(label in expected and label in predicted for expected, predicted, _ in records),
                "fp": sum(label not in expected and label in predicted for expected, predicted, _ in records),
                "fn": sum(label in expected and label not in predicted for expected, predicted, _ in records),
            }
            _, _, label_f1 = precision_recall_f1(label_counts)
            per_label_f1[label] = label_f1
        category_metrics[category_id]["multilabel_macro_f1"] = (
            sum(per_label_f1.values()) / len(per_label_f1) if per_label_f1 else None
        )
        category_metrics[category_id]["multilabel_per_label_f1"] = per_label_f1  # type: ignore[assignment]

    for category_id, counts in span_counts.items():
        precision, recall, f1 = precision_recall_f1(counts)
        category_metrics[category_id]["span_precision"] = precision
        category_metrics[category_id]["span_recall"] = recall
        category_metrics[category_id]["span_f1"] = f1

    for category_id, counts in edit_counts.items():
        precision, recall, f0_5 = precision_recall_f1(counts, beta_sq=0.25)
        category_metrics[category_id]["edit_precision"] = precision
        category_metrics[category_id]["edit_recall"] = recall
        category_metrics[category_id]["edit_f0_5"] = f0_5

    result = {
        "quality_success_count": success_count,
        "quality_failed_count": failed_count,
        "quality_total_count": total_count,
        "quality_success_rate": None if non_accuracy_categories else success_rate,
        "quality_score": (
            sum(category_quality_totals.values()) / total_count
            if total_count > 0 and not any("ungraded_generation" in scopes for scopes in non_accuracy_categories.values())
            else None
        ),
        "quality_ungraded_count": ungraded_count,
        "catastrophic_failure_count": len(catastrophic_failure_task_ids),
        "catastrophic_failure_task_ids": catastrophic_failure_task_ids,
        "scoring_error_count": len(scoring_error_task_ids),
        "scoring_error_task_ids": scoring_error_task_ids,
        "task_scores": task_scores,
        "category_results": category_metrics,
        "bucket_metrics": bucket_metrics,
        "decision_mode": {
            "runs": len(per_run_success_rate),
            "per_run_success_rate": per_run_success_rate,
            "min_success_rate": min(per_run_success_rate) if per_run_success_rate else 0.0,
            "max_success_rate": max(per_run_success_rate) if per_run_success_rate else 0.0,
            "mean_success_rate": sum(per_run_success_rate) / len(per_run_success_rate) if per_run_success_rate else 0.0,
            "stdev_success_rate": statistics.stdev(per_run_success_rate) if len(per_run_success_rate) > 1 else 0.0,
            "range": max(per_run_success_rate) - min(per_run_success_rate) if per_run_success_rate else 0.0,
        },
    }
    if non_accuracy_categories:
        # Success-rate stability and aggregate accuracy do not describe these
        # metrics. Counts still cover every attempted task for reliability.
        result["decision_mode"] = None

    return result
