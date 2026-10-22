"""One registry drives execution, validation and task-editor descriptions."""

from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from typing import Callable

from . import scorers as s


@dataclass(frozen=True)
class Scorer:
    id: str
    name: str
    description: str
    configuration: str
    example: dict
    evaluate: Callable
    graded: bool = True


def _tuple_score(function, output, task):
    score, reason = function(output, task)
    return s.ScoreDetail(score, reason, "generic")


def _numeric_tolerance(output, task):
    score, reason = s.score_numeric_tolerance(output, float(task["expected_answer"]), task.get("tolerance") or {"relative": 0.001, "absolute_floor": 0.005}, task)
    return s.ScoreDetail(score, reason, "generic")


# Each entry includes a real callable and a runnable reference example. There is
# no second list of supported IDs in the API or frontend.
SCORERS = {entry.id: entry for entry in (
    Scorer("numeric_exact", "Numeric answer", "Extracts the final number and compares it with the reference using a tight numeric tolerance.", "Expected answer: a finite number. Optional tolerance.relative and tolerance.absolute_floor.", {"expected_answer": 4, "response": "ANSWER: 4"}, partial(_tuple_score, s.score_numeric_exact)),
    Scorer("numeric_tolerance", "Numeric answer with tolerance", "Checks a numeric answer within configured relative tolerance; percentage handling is supported.", "Expected answer: a finite number. tolerance.relative and tolerance.relative_floor are fractions of the reference magnitude; legacy absolute_floor is also relative for this scorer.", {"expected_answer": 4, "response": "ANSWER: 4"}, _numeric_tolerance),
    Scorer("keyword_match", "Required keywords", "Passes when every expected keyword occurs in the output, ignoring case. This is substring matching, not semantic understanding.", "Expected answer: a keyword string or a nonempty JSON array of keywords.", {"expected_answer": ["blue", "sky"], "response": "The sky is blue."}, partial(_tuple_score, s.score_keyword_match)),
    Scorer("multiple_choice_exact", "Multiple choice", "Compares the extracted choice with the reference label.", "Expected answer: a choice label. scoring_config.allowed_choices can define custom labels; otherwise A–D.", {"expected_answer": "B", "response": "ANSWER: B"}, partial(_tuple_score, s.score_multiple_choice_exact)),
    Scorer("grounded_answer", "Answer and citation", "Requires the reference answer text and a citation from the task's approved list. Uses supplied context, with no retrieval service.", "Expected answer: text. scoring_config.valid_citations: nonempty array of source IDs such as source#p1.", {"expected_answer": "Paris", "scoring_config": {"valid_citations": ["source#p1"]}, "response": "Paris SOURCE: source#p1"}, s.score_grounded_answer_detail),
    Scorer("qa_f1", "Answer token F1", "Checks normalized exact match or token overlap against reference answers. Does not assess semantic equivalence.", "Expected answer: text. Optional accepted_answers, f1_threshold (default 0.8), require_citation and valid_citations in scoring_config.", {"expected_answer": "Paris", "response": "Paris"}, s.score_qa_f1_detail),
    Scorer("extractive_qa", "Strict JSON answer", "Requires exactly one JSON answer field and a normalized reference match; token F1 is diagnostic.", "Expected answer: text. Output must be {\"answer\":\"...\"}. Optional scoring_config.accepted_answers.", {"expected_answer": "Paris", "response": '{"answer":"Paris"}'}, s.score_extractive_qa_detail),
    Scorer("classification_label", "Single label", "Checks one label against a task-defined allowed label set.", "Expected answer: label text. scoring_config.allowed_labels: nonempty array; optional label_aliases object.", {"expected_answer": "positive", "scoring_config": {"allowed_labels": ["positive", "negative"]}, "response": "positive"}, s.score_classification_label_detail),
    Scorer("multilabel_f1", "Multiple labels", "Computes label-set F1 and passes at the configured threshold; unknown labels fail.", "Expected answer: JSON array of labels. scoring_config.allowed_labels required; f1_threshold defaults to 1.", {"expected_answer": ["science"], "scoring_config": {"allowed_labels": ["science", "sports"]}, "response": '["science"]'}, s.score_multilabel_f1_detail),
    Scorer("span_labels", "Labeled spans", "Matches labeled character spans by intersection-over-union and computes F1.", "Expected answer: JSON array of {start,end,label}. scoring_config.allowed_labels required; span_iou_threshold defaults to 0.5 and f1_threshold to 1.", {"expected_answer": [{"start": 0, "end": 5, "label": "place"}], "scoring_config": {"allowed_labels": ["place"]}, "response": '{"spans":[{"start":0,"end":5,"label":"place"}]}'}, s.score_span_labels_detail),
    Scorer("copy_edit_f05", "Reference edits", "Compares token edits against a corrected reference using F0.5, emphasizing precision.", "Expected answer: corrected text. scoring_config.source_text required; f0_5_threshold defaults to 1. Optional protected_values.", {"expected_answer": "The sky is blue.", "scoring_config": {"source_text": "The sky are blue."}, "response": "The sky is blue."}, s.score_copy_edit_f05_detail),
    Scorer("summary_reference", "Summary reference overlap", "Measures ROUGE token overlap and optional authored fact constraints. This does not grade writing quality or factual truth.", "Expected answer: reference text. Optional reference_summaries and min_rouge_1/2/l thresholds; min_rouge_l defaults to 0.5. No model-based similarity.", {"expected_answer": "The council approved the plan.", "response": "The council approved the plan."}, s.score_summary_reference_detail),
    Scorer("newsroom_facts", "Required facts and constraints", "Checks authored fact phrases, protected values, quotes, length and optional citations.", "Expected answer: text. scoring_config.required_facts required; optional protected_values, required_quotes, forbidden_phrases, min_words/max_words.", {"expected_answer": "The council approved the plan.", "scoring_config": {"required_facts": ["approved the plan"]}, "response": "The council approved the plan."}, s.score_newsroom_facts_detail),
    Scorer("atomic_claims", "Authored claim catalog", "Checks required and forbidden phrases from an authored claim catalog. Cannot discover arbitrary unsupported claims.", "Expected answer: text. scoring_config.claim_catalog: nonempty array of {id,text,status}; status may be required, optional or forbidden.", {"expected_answer": "The plan passed.", "scoring_config": {"claim_catalog": [{"id": "passed", "text": "plan passed", "status": "required"}]}, "response": "The plan passed."}, s.score_atomic_claims_detail),
    Scorer("editorial_rubric", "Rule-based content rubric", "Combines explicit reference overlap, fact, length, script and citation checks. No AI judge or prose-quality assessment.", "Expected answer: reference text. Optional scoring_config.required_facts, rubric_weights, pass_threshold, min_words/max_words and valid_citations.", {"expected_answer": "The council approved the plan.", "response": "The council approved the plan."}, s.score_editorial_rubric_detail),
    Scorer("editorial_ratings", "Reference rating agreement", "Compares numeric dimensions with fixed reference ratings. No model assigns the reference scores.", "Expected answer: JSON object of numeric ratings. Output uses the same keys; rating_min/max default to 1/5.", {"expected_answer": {"clarity": 4}, "response": '{"clarity":4}'}, s.score_rating_agreement_detail),
    Scorer("editorial_review", "Ungraded generation diagnostics", "Records word count and constraints without a quality score; retained for historical tasks only.", "Not selectable for new scored tasks.", {"expected_answer": "", "response": "Example text."}, s.score_generation_diagnostics_detail, graded=False),
)}


def catalog():
    return [{"id": entry.id, "name": entry.name, "description": entry.description,
             "configuration": entry.configuration, "example": entry.example,
             "expected_answer_type": type(entry.example["expected_answer"]).__name__}
            for entry in SCORERS.values() if entry.graded]


def fingerprint():
    """Identify both implementation and public contract, independent of paths."""
    from pathlib import Path
    digest = hashlib.sha256()
    for name in ("scorers.py", "registry.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def score_task_detail(response, task):
    entry = SCORERS.get(task.get("task_type"))
    if entry is None:
        raise NotImplementedError(f"No scorer registered for task_type={task.get('task_type')!r}")
    detail = entry.evaluate(response, task)
    if detail.score is None and detail.reason in {"no_answer_extracted", "no_number_extracted", "no_classification_label_extracted"}:
        return s.ScoreDetail(0.0, detail.reason, "format_error", detail.evidence)
    return detail


def validate_task(task, *, allow_ungraded=False):
    """Reject incomplete references/configuration before saving or inference."""
    entry = SCORERS.get(task.get("task_type"))
    if entry is None or (not entry.graded and not allow_ungraded):
        raise ValueError("Select a supported deterministic scorer for this task")
    cfg = task.get("scoring_config") or {}
    if not isinstance(cfg, dict):
        raise ValueError("scoring_config must be a JSON object")
    merged = {**cfg, **{key: value for key, value in task.items() if value is not None}}
    if any(key in merged for key in ("min_bertscore", "bertscore_model", "bertscore_device")):
        raise ValueError("Model-based BERTScore is not supported; use reference-overlap thresholds")
    expected = task.get("expected_answer")
    kind = type(entry.example["expected_answer"])
    if kind is dict and isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except ValueError:
            raise ValueError("Expected answer must be a JSON object") from None
    if entry.id in {"numeric_exact", "numeric_tolerance"}:
        if isinstance(expected, bool):
            raise ValueError("Expected answer must be a finite number")
        try:
            number = float(expected)
        except (ValueError, TypeError):
            raise ValueError("Expected answer must be a finite number") from None
        if not math.isfinite(number):
            raise ValueError("Expected answer must be a finite number")
    elif entry.id == "keyword_match":
        if not isinstance(expected, (str, list)) or not expected or (isinstance(expected, list) and any(not isinstance(v, str) or not v.strip() for v in expected)):
            raise ValueError("Expected keywords must be nonempty strings")
    elif not isinstance(expected, kind) or (kind is str and not expected.strip() and entry.graded):
        raise ValueError(f"Expected answer must be {kind.__name__}; see the scorer example")
    for key in ("allowed_labels", "allowed_choices", "accepted_answers", "reference_summaries", "valid_citations", "evidence_spans", "required_facts", "protected_values", "required_quotes", "forbidden_phrases", "claim_catalog"):
        if key in merged and not isinstance(merged[key], list):
            raise ValueError(f"{key} must be a JSON array")
    aliases = merged.get("label_aliases")
    if aliases is not None and (not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items())):
        raise ValueError("label_aliases must map text labels to text labels")
    if entry.id in {"classification_label", "multilabel_f1"}:
        allowed = merged.get("allowed_labels")
        if not allowed or any(not isinstance(label, str) or not label.strip() for label in allowed):
            raise ValueError("Configure a nonempty allowed_labels array")
        gold = [expected] if isinstance(expected, str) else expected
        if not gold or any(not isinstance(label, str) or s._normalize_classification_label(label) not in {s._normalize_classification_label(v) for v in allowed} for label in gold):
            raise ValueError("Expected labels must belong to allowed_labels")
    for key in ("min_words", "max_words"):
        if key in merged and (type(merged[key]) is not int or merged[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    if "max_words" in merged and merged["max_words"] < merged.get("min_words", 0):
        raise ValueError("max_words must be at least min_words")
    if entry.id == "grounded_answer" or merged.get("require_citation"):
        if not s._valid_citations(merged):
            raise ValueError("Configure a nonempty valid_citations list")
    if entry.id == "newsroom_facts" and not merged.get("required_facts"):
        raise ValueError("Configure required_facts; otherwise there are no facts to check")
    for key in ("f1_threshold", "f0_5_threshold", "span_iou_threshold", "min_fact_coverage", "min_required_coverage", "min_rouge_1", "min_rouge_2", "min_rouge_l", "pass_threshold", "min_script_ratio"):
        if key in merged:
            value = merged[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite number between 0 and 1")
    tolerance = task.get("tolerance") or {}
    if not isinstance(tolerance, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in tolerance.values()):
        raise ValueError("Tolerance values must be finite nonnegative numbers")
    weights = merged.get("rubric_weights")
    if weights is not None:
        if entry.id != "editorial_rubric":
            raise ValueError("rubric_weights requires the editorial_rubric scorer")
        if not isinstance(weights, dict) or any(key not in {"facts", "reference", "length", "script", "citation"} or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for key, value in weights.items()):
            raise ValueError("rubric_weights must contain nonnegative finite weights for known dimensions")
    try:
        # A malformed model answer may score zero; malformed task definitions
        # return an unavailable score or raise, and must never be accepted.
        detail = score_task_detail("", task)
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Invalid {entry.id} configuration: {exc}") from exc
    if entry.graded and detail.score is None:
        raise ValueError(f"Invalid {entry.id} configuration: {detail.reason}")
