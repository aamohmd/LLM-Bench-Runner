"""Scoring functions for benchmark task types.

Each function is pure: no I/O, no global state.
Return type is always (score: float | None, reason: str).

  score == 1.0  → correct
  score == 0.0  → wrong (counted failure)
  score == None → unavailable score; reason_code distinguishes intentional
                  ungraded generation from invalid scorer configuration

The reason string is always populated and is machine-readable enough to group by
in downstream analysis (e.g. "answer_tag" vs "fallback_extraction").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import math
import re
import unicodedata


@dataclass(frozen=True)
class ScoreDetail:
    """Stable machine-readable scoring evidence for approval aggregation."""

    score: float | None
    reason: str
    reason_code: str
    evidence: dict[str, object] = field(default_factory=dict)

    def as_tuple(self) -> tuple[float | None, str]:
        return self.score, self.reason


# ---------------------------------------------------------------------------
# Shared extraction utilities
# ---------------------------------------------------------------------------

# Primary: the format the prompt instructs the model to use.
# Captures the number immediately following "ANSWER:" — optional whitespace,
# optional leading sign, Unicode decimal digits, optional locale separators, optional % suffix.
_ANSWER_TAG_RE = re.compile(
    r"ANSWER\s*:\s*([-+]?\d[\d,\.٫٬]*%?)",
    re.IGNORECASE,
)

# Fallback: any standalone number (with optional sign, commas, decimals, %)
# anywhere in the text. "Standalone" means not followed by more digits or
# alphanumeric to avoid cutting off mid-word (e.g. "12.5x" → skip).
_GENERAL_NUMBER_RE = re.compile(
    r"[-+]?\s*[\$€£¥₹₽₩₺]?\s*\d[\d,\.٫٬]*%?(?!\d|\w)",
)

# Pass 1 — tag-based citation extraction.
# Accepts common keyword prefixes that LLMs use when citing a policy (EN, FR, AR):
#   CITATION, SOURCE, POLICY EXCERPT, EXCERPT, REFERENCE, REF, Policy, Citation, المرجع, الاستشهاد, Directive
# Optional surrounding square brackets are stripped from the captured ID.
_CITATION_TAG_RE = re.compile(
    r"(?:CITATION|SOURCE|POLICY(?:\s+EXCERPT)?|EXCERPT|REFERENCE|REF|المرجع|الاستشهاد|المصدر|Directive)\s*:\s*\[?([A-Za-z0-9_.-]+#p\d+)\]?",
    re.IGNORECASE,
)

# Pass 2 — fallback: any token of the form ID#pN anywhere in the output,
# with or without enclosing brackets.
_CITATION_BARE_RE = re.compile(
    r"\[?([A-Za-z0-9_.-]+#p\d+)\]?",
    re.IGNORECASE,
)


def _normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(
        str.maketrans({
            "‘": "'",
            "’": "'",
            "ʼ": "'",
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
        })
    )
    return re.sub(r"\s+", " ", text.strip().casefold())


def _valid_citations(task: dict) -> set[str]:
    cfg = task.get("scoring_config") if isinstance(task.get("scoring_config"), dict) else {}
    explicit = task.get("valid_citations") or cfg.get("valid_citations")
    if isinstance(explicit, list):
        return {_normalized_text(value) for value in explicit}
    evidence = task.get("evidence_spans") or cfg.get("evidence_spans") or []
    return {
        _normalized_text(f"{span['source_id']}#p{span['page']}")
        for span in evidence
        if isinstance(span, dict) and span.get("source_id") and isinstance(span.get("page"), int)
    }


def _citation_detail(model_output: str, task: dict) -> tuple[bool, str, str]:
    """Two-pass citation verification.

    Pass 1: Look for an explicit keyword tag (CITATION:, SOURCE:, POLICY EXCERPT:, etc.)
            followed by an ID#pN citation, with optional surrounding brackets.
    Pass 2: Scan the full output for any bare ID#pN token and cross-check it
            against the task's valid citation set.  This catches the very common
            model pattern of embedding the excerpt ID inline, e.g.:
              ``Policy Excerpt: [BAM-AML-001#p1] Transactions …``

    Returns:
        (True,  "valid_citation")            — a recognised citation was found
        (False, "missing_citation")          — no citation-like token at all
        (False, "invalid_citation")          — tokens found but none is valid
        (False, "no_valid_citations_defined")— task carries no citation spec
    """
    valid = _valid_citations(task)
    if not valid:
        return False, "no_valid_citations_defined", "citation_spec_missing"

    # Pass 1: tag-anchored extraction
    tag_hits = {_normalized_text(v) for v in _CITATION_TAG_RE.findall(model_output)}
    if tag_hits & valid:
        return True, "valid_citation", "valid_citation"

    # Pass 2: bare-token fallback (catches inline [ID#pN] and plain ID#pN)
    bare_hits = {_normalized_text(v) for v in _CITATION_BARE_RE.findall(model_output)}
    if bare_hits & valid:
        return True, "valid_citation", "valid_citation"

    distractors = {_normalized_text(value) for value in task.get("distractor_citations", [])}
    if (tag_hits | bare_hits) & distractors:
        return False, "invalid_citation", "cited_distractor_source"

    if not tag_hits and not bare_hits:
        return False, "missing_citation", "missing_citation"
    return False, "invalid_citation", "invalid_citation"


def _citation_status(model_output: str, task: dict) -> tuple[bool, str]:
    """Compatibility wrapper for existing tuple-based scorers."""
    ok, reason, _ = _citation_detail(model_output, task)
    return ok, reason


def _normalize_number(raw: str) -> tuple[float | None, bool]:
    """Normalize an extracted number string to a float in decimal-fraction form.

    Steps (in order):
    1. Translate any Unicode decimal digits to ASCII (0-9).
    2. Strip currency symbols and commas.
    3. If value ends with %: divide by 100 (canonical decimal form).
    4. Parse to float.

    Returns (value, is_pct) where value is None if parsing fails.
    """
    raw_str = "".join(
        str(unicodedata.decimal(char)) if unicodedata.category(char) == "Nd" else char
        for char in unicodedata.normalize("NFKC", str(raw))
    ).translate(str.maketrans({"٫": ".", "٬": ","}))
    # Remove currency symbols, commas, and whitespace, preserving the sign
    cleaned = re.sub(r'[\$€£¥₹₽₩₺\s,]', '', raw_str)
    is_pct = cleaned.endswith("%")
    if is_pct:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None, False
    if is_pct:
        value /= 100.0
    return value, is_pct


def _extract_number(text: str) -> tuple[float | None, str, bool]:
    """Two-pass number extraction. Returns (value, extraction_path_tag, was_percent).

    Pass 1: look for ANSWER: <number> (format-compliant).
    Pass 2: take the last standalone number in the full text (fallback).

    Returns (None, "no_number_extracted", False) if nothing parseable is found.
    The extraction_path_tag is always a non-empty string, for use in
    downstream reason-string analysis.
    """
    normalized_input = unicodedata.normalize("NFKC", text)

    # Pass 1 — primary: ANSWER: tag
    m = _ANSWER_TAG_RE.search(normalized_input)
    if m:
        value, is_pct = _normalize_number(m.group(1))
        if value is not None:
            return value, "answer_tag", is_pct

    # Pass 2 — fallback: last number in full text
    all_matches = _GENERAL_NUMBER_RE.findall(normalized_input)
    if all_matches:
        value, is_pct = _normalize_number(all_matches[-1])
        if value is not None:
            return value, "fallback_extraction", is_pct

    return None, "no_number_extracted", False


# ---------------------------------------------------------------------------
# Multiple-choice (MMLU) scorer
# ---------------------------------------------------------------------------

# Primary: explicit ANSWER: tag followed by a single letter A-D.
# Case-insensitive tag; the captured letter may be lower- or uppercase.
_ANSWER_LETTER_TAG_RE = re.compile(
    r"ANSWER\s*:\s*([A-Da-d])",
    re.IGNORECASE,
)

# Fallback: first standalone UPPERCASE A-D token. Lowercase letters are
# deliberately excluded — a lone lowercase "a"/"b" is ambiguous (e.g. the
# article "a"), while an uppercase A-D token is almost always an answer letter.
_STANDALONE_LETTER_RE = re.compile(r"(?<![\w])[A-D](?![\w])")


def score_multiple_choice_exact(
    model_output: str,
    task: dict,
) -> tuple[float | None, str]:
    """Score a task-defined multiple-choice answer by exact normalized match.

    ``allowed_choices`` (or the keys of a ``choices`` mapping) may use any
    labels.  A-D remains the backwards-compatible default.

    Returns:
        (1.0, "match via answer_tag")       — tag found, letter matches
        (1.0, "match via fallback_extraction") — fallback letter matches
        (0.0, "wrong_answer: got X expected Y") — letter extracted, mismatch
        (None, "no_answer_extracted")       — no letter extractable
    """
    expected = _normalized_text(task.get("expected_answer", ""))
    if not expected:
        return None, "no_expected_answer"

    cfg = task.get("scoring_config") or {}
    raw_allowed = task.get("allowed_choices") or cfg.get("allowed_choices")
    if raw_allowed is None and isinstance(task.get("choices"), dict):
        raw_allowed = list(task["choices"])
    if raw_allowed is None:
        raw_allowed = ["A", "B", "C", "D"]
    if isinstance(raw_allowed, (str, bytes)) or not isinstance(raw_allowed, (list, tuple, set)):
        return None, "invalid_allowed_choices"
    allowed = {_normalized_text(value): str(value).strip() for value in raw_allowed}
    allowed.pop("", None)
    if expected not in allowed:
        return None, "expected_answer_not_in_allowed_choices"

    parsed = _extract_json_object(model_output)
    if isinstance(parsed, dict) and "answer" in parsed:
        got = _normalized_text(parsed["answer"])
        if got in allowed:
            if got == expected:
                return 1.0, "match via structured_json"
            return 0.0, f"wrong_answer: got {allowed[got]} expected {allowed[expected]}"

    legacy_choices = set(allowed) == {"a", "b", "c", "d"}
    if legacy_choices:
        m = _ANSWER_LETTER_TAG_RE.search(model_output)
        if m:
            got = _normalized_text(m.group(1))
            if got == expected:
                return 1.0, "match via answer_tag"
            return 0.0, f"wrong_answer: got {allowed[got]} expected {allowed[expected]}"

    generic_tag = re.search(r"^ANSWER\s*:\s*(.+?)\s*$", model_output, re.IGNORECASE | re.MULTILINE)
    if generic_tag:
        got = _normalized_text(generic_tag.group(1))
        if got in allowed:
            if got == expected:
                return 1.0, "match via answer_tag"
            return 0.0, f"wrong_answer: got {allowed[got]} expected {allowed[expected]}"

    exact_lines = {
        _normalized_text(line)
        for line in model_output.splitlines()
        if _normalized_text(line) in allowed
    }
    if len(exact_lines) == 1:
        got = exact_lines.pop()
        if got == expected:
            return 1.0, "match via fallback_extraction"
        return 0.0, f"wrong_answer: got {allowed[got]} expected {allowed[expected]}"

    # Preserve the safe legacy A-D fallback for prose responses. Lowercase
    # articles are deliberately not treated as answers.
    if legacy_choices:
        m = _STANDALONE_LETTER_RE.search(model_output)
        if m:
            got = _normalized_text(m.group(0))
            if got == expected:
                return 1.0, "match via fallback_extraction"
            return 0.0, f"wrong_answer: got {allowed[got]} expected {allowed[expected]}"

    return None, "no_answer_extracted"


def score_numeric_exact(
    model_output: str,
    task: dict,
) -> tuple[float | None, str]:
    """Score an exact numeric answer (GSM8K-style) with tight tolerance.

    Extraction reuses the two-pass _extract_number (ANSWER: tag preferred,
    standalone-number fallback). A numeric answer like "23" or "23.0" must
    match the gold; a near-miss (23.01 vs 23) fails.

    tolerance (task-provided or default): relative 1e-4, no scaled candidates —
    GSM8K golds are exact decimal values, and 100x scaling is never legitimate.

    Returns:
        (1.0, "exact_match via <extraction_tag>") — within tolerance
        (0.0, "outside_tolerance: got X expected Y") — wrong but parseable
        (None, "no_number_extracted")             — extraction failed
    """
    expected = float(task["expected_answer"])
    extracted, extraction_tag, _ = _extract_number(model_output)
    if extracted is None:
        return None, "no_number_extracted"

    tolerance = task.get("tolerance") or {}
    rel = tolerance.get("relative", 1e-4)
    floor = tolerance.get("absolute_floor", 1e-4)
    threshold = max(abs(expected) * rel, floor)

    delta = abs(extracted - expected)
    if delta <= threshold:
        return 1.0, f"exact_match via {extraction_tag} | delta: {delta}"
    return (
        0.0,
        f"outside_tolerance: got {extracted} expected {expected} | delta: {delta}",
    )


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def score_keyword_match(
    model_output: str,
    task: dict,
) -> tuple[float | None, str]:
    """Score by checking if expected keywords exist in the output."""
    expected = task.get("expected_keywords") or task.get("expected_answer")
    if not expected:
        return None, "no_expected_keywords"
    
    if isinstance(expected, list):
        for kw in expected:
            if str(kw).lower() not in model_output.lower():
                return 0.0, f"missing_keyword: {kw}"
        return 1.0, "all_keywords_found"
    else:
        if str(expected).lower() in model_output.lower():
            return 1.0, "keyword_found"
        return 0.0, f"missing_keyword: {expected}"


def score_grounded_answer_detail(
    model_output: str,
    task: dict,
) -> ScoreDetail:
    """Require a deterministic answer match and a task-approved citation."""
    expected = task.get("expected_answer")
    if not isinstance(expected, str) or not expected.strip():
        return ScoreDetail(None, "no_expected_answer", "expected_answer_missing")
    if _normalized_text(expected) not in _normalized_text(model_output):
        return ScoreDetail(0.0, "wrong_or_missing_answer", "wrong_answer")
    citation_ok, citation_reason, citation_code = _citation_detail(model_output, task)
    if not citation_ok:
        return ScoreDetail(0.0, citation_reason, citation_code, {"answer_correct": True})
    return ScoreDetail(1.0, "answer_and_valid_citation", "answer_and_valid_citation", {"answer_correct": True, "citation_correct": True})


def score_grounded_answer(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_grounded_answer_detail(model_output, task).as_tuple()


def score_numeric_tolerance(
    model_output: str,
    expected_answer: float,
    tolerance: dict,
    task: dict | None = None,
) -> tuple[float | None, str]:
    """Score a numeric answer against the gold value using relative tolerance.

    `expected_answer` arrives pre-normalized from the task suite file (decimal
    fraction, no currency symbols). Normalization is only applied to the
    extracted model output.

    tolerance dict keys:
        relative       — fraction of |expected_answer| used as threshold
        relative_floor — additional fraction of |expected_answer| as a floor,
                         so the tolerance scales with the gold's magnitude.
                         (Legacy key absolute_floor is accepted but applied
                         RELATIVELY, matching its original 0.005 default.)

    Returns:
        (1.0, "match via <extraction_tag>")     — within tolerance
        (0.0, "outside_tolerance: got X expected Y") — wrong but parseable
        (None, "no_number_extracted")            — extraction failed
    """
    extracted, extraction_tag, was_percent = _extract_number(model_output)

    if extracted is None:
        return None, "no_number_extracted"

    expected_abs = abs(expected_answer)
    rel = tolerance.get("relative", 0.001)
    floor_factor = tolerance.get("relative_floor", tolerance.get("absolute_floor", 0.005))
    
    # Continuous scale floor: scales continuously with order of magnitude without cliff discontinuities
    if expected_abs > 0:
        log_order = math.floor(math.log10(max(expected_abs, 1e-6)))
        continuous_floor = 10.0 ** (log_order - 3)
    else:
        continuous_floor = 1e-6

    threshold = max(expected_abs * rel, expected_abs * floor_factor, continuous_floor, 1e-6)

    # Check whether percentage / candidate scaling is permissible
    task_allows_percent = False
    if task is None:
        task_allows_percent = True
    elif was_percent:
        task_allows_percent = True
    elif task.get("pct_flag") or task.get("bucket") == "numeric_reasoning":
        task_allows_percent = True
    elif any(kw in task.get("prompt", "").lower() for kw in ("percent", "%", "margin", "yield", "rate", "bps")):
        task_allows_percent = True
    elif "%" in model_output:
        task_allows_percent = True

    candidates = {"identity": extracted}
    if task_allows_percent:
        candidates["scaled_down"] = extracted / 100.0
        candidates["scaled_up"] = extracted * 100.0

    scaled_threshold = threshold * 5

    for scale_name, candidate in candidates.items():
        delta = abs(candidate - expected_answer)
        effective_threshold = (
            scaled_threshold
            if (scale_name != "identity" or was_percent)
            else threshold
        )
        if delta <= effective_threshold:
            reason = f"match via {extraction_tag}"
            if scale_name != "identity":
                reason += f" (non_identity_match: {scale_name}, got {extracted} expected {expected_answer})"
            elif was_percent:
                reason += f" (percent_rounding_match, got {extracted} expected {expected_answer})"
            reason += f" | delta: {delta}"
            return 1.0, reason

    return (
        0.0,
        f"outside_tolerance: got {extracted} expected {expected_answer} | delta: {abs(extracted - expected_answer)}",
    )


def _extract_json_object(model_output: str) -> dict | None:
    """Extract a JSON object dict from model output.

    Handles markdown fences anywhere in the text (models often prefix JSON
    with prose like "Here is the extracted data:") and falls back to the
    first '{' ... last '}' span. Returns None if no valid object is found.
    """
    import json

    cleaned = model_output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            candidate = cleaned
        else:
            candidate = cleaned[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strict_json_object(model_output: str) -> dict | None:
    """Parse a complete object, rejecting duplicate keys and non-JSON numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(value)

    try:
        value = json.loads(model_output, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def score_extractive_qa_detail(model_output: str, task: dict) -> ScoreDetail:
    """Exact short-answer matching with aliases; token F1 is diagnostic only."""
    answers = _config_value(task, "accepted_answers") or [task.get("expected_answer")]
    if _config_value(task, "is_unanswerable", False):
        answers = [task.get("expected_answer")]
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not _normalize_qa_text(a) for a in answers):
        raise ValueError("extractive QA requires nonempty reference strings")
    parsed = _strict_json_object(model_output)
    if parsed is None or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
        return ScoreDetail(0.0, "expected_single_json_answer", "format_error")
    answer = parsed["answer"]
    exact = any(_normalize_qa_text(answer) == _normalize_qa_text(gold) for gold in answers)
    return ScoreDetail(float(exact), "answer_match" if exact else "answer_mismatch", "extractive_qa", {
        "exact_match": exact,
        "token_f1": max(_token_f1(answer, gold) for gold in answers),
        "is_unanswerable": bool(_config_value(task, "is_unanswerable", False)),
    })


def score_extractive_qa(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_extractive_qa_detail(model_output, task).as_tuple()


def score_rating_agreement_detail(model_output: str, task: dict) -> ScoreDetail:
    """Compare arbitrary numeric dimensions to published reference ratings."""
    expected = task.get("expected_answer")
    gold = _strict_json_object(expected) if isinstance(expected, str) else expected
    low = float(_config_value(task, "rating_min", 1))
    high = float(_config_value(task, "rating_max", 5))

    def valid(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high and math.isfinite(value)

    if not math.isfinite(low) or not math.isfinite(high) or high <= low or not isinstance(gold, dict) or not gold or not all(valid(v) for v in gold.values()):
        raise ValueError("invalid reference ratings or rating scale")
    predicted = _strict_json_object(model_output)
    if predicted is None or set(predicted) != set(gold) or not all(valid(v) for v in predicted.values()):
        return ScoreDetail(0.0, "invalid_rating_object", "format_error")
    errors = {key: abs(predicted[key] - value) for key, value in gold.items()}
    mae = sum(errors.values()) / len(errors)
    return ScoreDetail(1 - mae / (high - low), "numeric_rating_agreement", "rating_agreement", {
        "rating_mae": mae, "dimension_absolute_errors": errors,
        "rating_min": low, "rating_max": high,
        "normalized_quality_score": 1 - mae / (high - low),
    })


def score_rating_agreement(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_rating_agreement_detail(model_output, task).as_tuple()


def score_generation_diagnostics_detail(model_output: str, task: dict) -> ScoreDetail:
    """Record measurable constraints without inventing semantic writing grades."""
    minimum = _config_value(task, "min_words", 0)
    maximum = _config_value(task, "max_words")
    if type(minimum) is not int or minimum < 0 or (maximum is not None and (type(maximum) is not int or maximum < minimum)):
        raise ValueError("invalid word-count bounds")
    words = len(_editorial_tokens(model_output))
    return ScoreDetail(None, "semantic_quality_not_graded", "ungraded_generation", {
        "word_count": words, "min_words": minimum, "max_words": maximum,
        "nonempty": bool(model_output.strip()),
        "length_compliant": words >= minimum and (maximum is None or words <= maximum),
    })


def score_generation_diagnostics(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_generation_diagnostics_detail(model_output, task).as_tuple()


# ---------------------------------------------------------------------------
# Reference-based deterministic scorers
# ---------------------------------------------------------------------------

_TEXT_ANSWER_TAG_RE = re.compile(
    r"^(?:ANSWER|R[ÉE]PONSE)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LABEL_TAG_RE = re.compile(
    r"^(?:ANSWER|LABEL|CLASS|CLASSE|CAT[ÉE]GORIE)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_qa_text(value: object) -> str:
    """Apply deterministic, accent-preserving normalization for French QA."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(
        " " if unicodedata.category(char).startswith(("P", "S")) else char
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def _qa_tokens(value: object) -> list[str]:
    normalized = _normalize_qa_text(value)
    return normalized.split() if normalized else []


def _token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _qa_tokens(prediction)
    reference_tokens = _qa_tokens(reference)
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_qa_answer(model_output: str) -> str:
    tagged = _TEXT_ANSWER_TAG_RE.search(model_output)
    if tagged:
        return tagged.group(1).strip()
    without_citations = re.sub(
        r"^(?:CITATION|SOURCE|R[ÉE]F[ÉE]RENCE)\s*:.*$",
        "",
        model_output,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return without_citations.strip()


def score_qa_f1_detail(model_output: str, task: dict) -> ScoreDetail:
    """Score French extractive QA, refusal behavior, and optional citations.

    The evidence retains EM and token-F1 independently. The binary task score
    applies the configured F1 threshold so it remains compatible with approval
    aggregation, which treats only 1.0 as a passing task.
    """
    raw_answers = _config_value(task, "accepted_answers")
    if raw_answers is None:
        expected = task.get("expected_answer")
        raw_answers = [expected] if expected is not None else []
    accepted_answers = [str(value) for value in raw_answers if str(value).strip()]
    is_unanswerable = bool(_config_value(task, "is_unanswerable", False)) or not accepted_answers
    answer = _extract_qa_answer(model_output)
    require_citation = bool(_config_value(task, "require_citation", False))
    citation_ok = True
    citation_reason = "citation_not_required"
    if require_citation:
        citation_ok, citation_reason, _ = _citation_detail(model_output, task)

    if is_unanswerable:
        phrases = _config_value(task, "refusal_phrases") or [
            "information non fournie",
            "impossible de répondre",
            "aucune réponse",
            "le document ne permet pas",
            "not enough information",
            "cannot answer",
        ]
        normalized_output = _normalize_qa_text(answer)
        refusal_correct = bool(normalized_output) and any(
            _normalize_qa_text(phrase) in normalized_output for phrase in phrases
        )
        passed = refusal_correct and citation_ok
        if not citation_ok:
            reason_code = "qa_citation_failed"
        elif refusal_correct:
            reason_code = "qa_refusal_correct"
        else:
            reason_code = "qa_should_refuse"
        return ScoreDetail(
            1.0 if passed else 0.0,
            "correct_refusal" if passed else "answer_given_for_unanswerable_question",
            reason_code,
            {
                "answerable": False,
                "refusal_correct": refusal_correct,
                "citation_ok": citation_ok,
                "citation_reason": citation_reason,
            },
        )

    exact_match = any(
        _normalize_qa_text(answer) == _normalize_qa_text(reference)
        for reference in accepted_answers
    )
    token_f1 = max(_token_f1(answer, reference) for reference in accepted_answers)
    threshold = float(_config_value(task, "f1_threshold", 0.8))
    answer_ok = exact_match or token_f1 >= threshold
    passed = answer_ok and citation_ok
    if not citation_ok:
        reason_code = "qa_citation_failed"
    elif not answer_ok:
        reason_code = "qa_answer_failed"
    else:
        reason_code = "qa_answer_passed"
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"qa_{'passed' if passed else 'failed'} (em={int(exact_match)}, f1={token_f1:.3f})",
        reason_code,
        {
            "answerable": True,
            "exact_match": exact_match,
            "token_f1": round(token_f1, 6),
            "f1_threshold": threshold,
            "citation_ok": citation_ok,
            "citation_reason": citation_reason,
        },
    )


def score_qa_f1(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_qa_f1_detail(model_output, task).as_tuple()


def _normalize_classification_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[\s-]+", "_", text)
    return re.sub(r"[^\w]", "", text, flags=re.UNICODE).strip("_")


def score_classification_label_detail(model_output: str, task: dict) -> ScoreDetail:
    """Score one classification item and retain labels for macro-F1 aggregation."""
    expected = _normalize_classification_label(task.get("expected_answer", ""))
    allowed = [
        _normalize_classification_label(value)
        for value in (_config_value(task, "allowed_labels", []) or [])
    ]
    if not expected:
        return ScoreDetail(None, "no_expected_answer", "expected_answer_missing")
    if not allowed:
        return ScoreDetail(None, "no_allowed_labels", "allowed_labels_missing")

    aliases = {
        _normalize_classification_label(alias): _normalize_classification_label(canonical)
        for alias, canonical in (_config_value(task, "label_aliases", {}) or {}).items()
    }
    tagged = _LABEL_TAG_RE.search(model_output)
    candidate = _normalize_classification_label(tagged.group(1)) if tagged else ""
    candidate = aliases.get(candidate, candidate)
    if candidate not in allowed:
        exact_label_lines = {
            _normalize_classification_label(line)
            for line in model_output.splitlines()
            if _normalize_classification_label(line) in allowed
        }
        candidate = exact_label_lines.pop() if len(exact_label_lines) == 1 else ""
        candidate = aliases.get(candidate, candidate)
    if candidate not in allowed:
        return ScoreDetail(
            None,
            "no_classification_label_extracted",
            "classification_label_missing",
            {"gold_label": expected, "predicted_label": "__invalid__"},
        )
    passed = candidate == expected
    return ScoreDetail(
        1.0 if passed else 0.0,
        "classification_match" if passed else f"classification_mismatch: got {candidate} expected {expected}",
        "classification_match" if passed else "classification_mismatch",
        {"gold_label": expected, "predicted_label": candidate},
    )


def score_classification_label(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_classification_label_detail(model_output, task).as_tuple()


def _fact_alternatives(raw_fact: object) -> tuple[str, list[str]]:
    if isinstance(raw_fact, dict):
        fact_id = str(raw_fact.get("id") or raw_fact.get("fact") or "fact")
        raw_alternatives = raw_fact.get("alternatives") or [raw_fact.get("fact", "")]
        return fact_id, [str(value) for value in raw_alternatives if str(value).strip()]
    value = str(raw_fact)
    return value, [value] if value.strip() else []


def _normalize_verbatim(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def score_newsroom_facts_detail(model_output: str, task: dict) -> ScoreDetail:
    """Apply hard, deterministic gates to a newsroom generation response.

    This scorer deliberately does not claim to assess prose quality or discover
    arbitrary unsupported claims. Those require a separately calibrated judge
    and human audit. It certifies only task-authored facts and invariants.
    """
    cfg = task.get("scoring_config") if isinstance(task.get("scoring_config"), dict) else {}
    normalized_output = _normalized_text(model_output)
    fact_specs = [
        _fact_alternatives(value)
        for value in (task.get("required_facts") or cfg.get("required_facts") or [])
    ]
    matched_fact_ids = [
        fact_id
        for fact_id, alternatives in fact_specs
        if alternatives and any(_normalized_text(value) in normalized_output for value in alternatives)
    ]
    fact_coverage = len(matched_fact_ids) / len(fact_specs) if fact_specs else 1.0
    min_fact_coverage = float(task.get("min_fact_coverage") or cfg.get("min_fact_coverage") or 1.0)

    protected_values = [
        str(value) for value in (task.get("protected_values") or cfg.get("protected_values") or [])
        if str(value).strip()
    ]
    missing_values = [value for value in protected_values if _normalized_text(value) not in normalized_output]

    required_quotes = [
        str(value) for value in (task.get("required_quotes") or cfg.get("required_quotes") or [])
        if str(value).strip()
    ]
    verbatim_output = _normalize_verbatim(model_output)
    missing_quotes = [quote for quote in required_quotes if _normalize_verbatim(quote) not in verbatim_output]

    forbidden = [
        str(value) for value in (task.get("forbidden_phrases") or cfg.get("forbidden_phrases") or [])
        if str(value).strip()
    ]
    forbidden_hits = [phrase for phrase in forbidden if _normalized_text(phrase) in normalized_output]

    words = re.findall(r"[^\W_]+(?:[’'][^\W_]+)*", model_output, flags=re.UNICODE)
    min_words = int(task.get("min_words") or cfg.get("min_words") or 1)
    max_words = int(task.get("max_words") or cfg.get("max_words") or 10_000)
    length_ok = min_words <= len(words) <= max_words

    require_citation = bool(task.get("require_citation") or cfg.get("require_citation"))
    citation_ok = True
    citation_reason = "citation_not_required"
    if require_citation:
        citation_ok, citation_reason, _ = _citation_detail(model_output, task)

    passed = (
        fact_coverage >= min_fact_coverage
        and not missing_values
        and not missing_quotes
        and not forbidden_hits
        and length_ok
        and citation_ok
    )
    evidence = {
        "fact_coverage": round(fact_coverage, 6),
        "matched_fact_ids": matched_fact_ids,
        "missing_fact_ids": [fact_id for fact_id, _ in fact_specs if fact_id not in matched_fact_ids],
        "protected_values_ok": not missing_values,
        "missing_protected_values": missing_values,
        "quote_fidelity_ok": not missing_quotes,
        "missing_quotes": missing_quotes,
        "forbidden_hits": forbidden_hits,
        "word_count": len(words),
        "length_ok": length_ok,
        "citation_ok": citation_ok,
        "citation_reason": citation_reason,
    }
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"newsroom_facts_{'passed' if passed else 'failed'} ({fact_coverage:.2f} fact coverage)",
        "newsroom_facts_passed" if passed else "newsroom_facts_failed",
        evidence,
    )


def score_newsroom_facts(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_newsroom_facts_detail(model_output, task).as_tuple()


def _config_value(task: dict, key: str, default: object = None) -> object:
    if key in task:
        return task[key]
    cfg = task.get("scoring_config")
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def _ngram_counts(tokens: list[str], size: int) -> Counter[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return Counter()
    return Counter(tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1))


def _overlap_f1(predicted: Counter, reference: Counter) -> float:
    predicted_total = sum(predicted.values())
    reference_total = sum(reference.values())
    if not predicted_total or not reference_total:
        return float(predicted_total == reference_total)
    overlap = sum((predicted & reference).values())
    if not overlap:
        return 0.0
    precision = overlap / predicted_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    predicted_tokens = _qa_tokens(prediction)
    reference_tokens = _qa_tokens(reference)
    rouge_1 = _overlap_f1(_ngram_counts(predicted_tokens, 1), _ngram_counts(reference_tokens, 1))
    rouge_2 = _overlap_f1(_ngram_counts(predicted_tokens, 2), _ngram_counts(reference_tokens, 2))
    if not predicted_tokens or not reference_tokens:
        rouge_l = float(predicted_tokens == reference_tokens)
    else:
        lcs = _lcs_length(predicted_tokens, reference_tokens)
        precision = lcs / len(predicted_tokens)
        recall = lcs / len(reference_tokens)
        rouge_l = 2 * precision * recall / (precision + recall) if lcs else 0.0
    return {"rouge_1_f1": rouge_1, "rouge_2_f1": rouge_2, "rouge_l_f1": rouge_l}


def score_summary_reference_detail(model_output: str, task: dict) -> ScoreDetail:
    """Score a summary/headline against frozen references and hard fact gates."""
    if any(_config_value(task, key) is not None for key in ("min_bertscore", "bertscore_model", "bertscore_device")):
        raise ValueError("Model-based BERTScore is unsupported; configure ROUGE reference-overlap thresholds")
    raw_references = _config_value(task, "reference_summaries")
    if raw_references is None:
        raw_references = task.get("accepted_answers")
    if raw_references is None:
        raw_references = [task.get("expected_answer")]
    if isinstance(raw_references, str):
        raw_references = [raw_references]
    references = [str(value) for value in (raw_references or []) if str(value).strip()]
    if not references:
        return ScoreDetail(None, "no_reference_summary", "summary_reference_missing")

    prediction = _extract_qa_answer(model_output)
    scored_references = [(reference, _rouge_scores(prediction, reference)) for reference in references]
    reference, metrics = max(scored_references, key=lambda item: item[1]["rouge_l_f1"])
    normalized_output = _normalized_text(prediction)

    fact_specs = [
        _fact_alternatives(value)
        for value in (_config_value(task, "required_facts", []) or [])
    ]
    matched_fact_ids = [
        fact_id
        for fact_id, alternatives in fact_specs
        if alternatives and any(_normalized_text(value) in normalized_output for value in alternatives)
    ]
    fact_coverage = len(matched_fact_ids) / len(fact_specs) if fact_specs else 1.0
    protected_values = [
        str(value) for value in (_config_value(task, "protected_values", []) or [])
        if str(value).strip()
    ]
    missing_values = [value for value in protected_values if _normalized_text(value) not in normalized_output]

    min_rouge_1 = float(_config_value(task, "min_rouge_1", 0.0))
    min_rouge_2 = float(_config_value(task, "min_rouge_2", 0.0))
    min_rouge_l = float(_config_value(task, "min_rouge_l", 0.5))
    min_fact_coverage = float(_config_value(task, "min_fact_coverage", 1.0))
    passed = (
        metrics["rouge_1_f1"] >= min_rouge_1
        and metrics["rouge_2_f1"] >= min_rouge_2
        and metrics["rouge_l_f1"] >= min_rouge_l
        and fact_coverage >= min_fact_coverage
        and not missing_values
    )
    evidence: dict[str, object] = {
        **{key: round(value, 6) for key, value in metrics.items()},
        "fact_coverage": round(fact_coverage, 6),
        "matched_fact_ids": matched_fact_ids,
        "missing_fact_ids": [fact_id for fact_id, _ in fact_specs if fact_id not in matched_fact_ids],
        "protected_values_ok": not missing_values,
        "missing_protected_values": missing_values,
    }
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"summary_reference_{'passed' if passed else 'failed'} (rougeL={metrics['rouge_l_f1']:.3f})",
        "summary_reference_passed" if passed else "summary_reference_failed",
        evidence,
    )


def score_summary_reference(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_summary_reference_detail(model_output, task).as_tuple()


def _parse_label_list(model_output: str) -> list[str] | None:
    parsed = _extract_json_object(model_output)
    if isinstance(parsed, dict) and isinstance(parsed.get("labels"), list):
        return [str(value) for value in parsed["labels"]]
    match = re.search(
        r"^(?:LABELS|ÉTIQUETTES|ETIQUETTES|CATÉGORIES|CATEGORIES)\s*:\s*(.+?)\s*$",
        model_output,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return [part.strip() for part in re.split(r"[,;|]", match.group(1)) if part.strip()]
    normalized = model_output.strip()
    return [normalized] if normalized else None


def score_multilabel_f1_detail(model_output: str, task: dict) -> ScoreDetail:
    raw_expected = _config_value(task, "expected_labels", task.get("expected_answer"))
    if isinstance(raw_expected, str):
        raw_expected = [raw_expected]
    expected = {_normalize_classification_label(value) for value in (raw_expected or [])}
    allowed = {
        _normalize_classification_label(value)
        for value in (_config_value(task, "allowed_labels", []) or [])
    }
    if not expected:
        return ScoreDetail(None, "no_expected_labels", "expected_labels_missing")
    if not allowed or not expected <= allowed:
        return ScoreDetail(None, "invalid_allowed_labels", "allowed_labels_invalid")
    aliases = {
        _normalize_classification_label(alias): _normalize_classification_label(canonical)
        for alias, canonical in (_config_value(task, "label_aliases", {}) or {}).items()
    }
    raw_predicted = _parse_label_list(model_output)
    if raw_predicted is None:
        raw_predicted = []
    normalized_predictions = [
        aliases.get(_normalize_classification_label(value), _normalize_classification_label(value))
        for value in raw_predicted
    ]
    invalid = sorted({value for value in normalized_predictions if value not in allowed})
    predicted = {value for value in normalized_predictions if value in allowed}
    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected) + len(invalid)
    false_negative = len(expected - predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 1.0
    threshold = float(_config_value(task, "f1_threshold", 1.0))
    passed = f1 >= threshold and not invalid
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"multilabel_{'passed' if passed else 'failed'} (f1={f1:.3f})",
        "multilabel_passed" if passed else "multilabel_failed",
        {
            "expected_labels": sorted(expected),
            "predicted_labels": sorted(predicted),
            "allowed_labels": sorted(allowed),
            "invalid_labels": invalid,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": round(f1, 6),
        },
    )


def score_multilabel_f1(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_multilabel_f1_detail(model_output, task).as_tuple()


def _valid_span(raw_span: object, allowed: set[str]) -> dict[str, object] | None:
    if not isinstance(raw_span, dict):
        return None
    label = _normalize_classification_label(raw_span.get("label", ""))
    start = raw_span.get("start")
    end = raw_span.get("end")
    if label not in allowed or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return None
    return {"label": label, "start": start, "end": end}


def _span_iou(left: dict[str, object], right: dict[str, object]) -> float:
    intersection = max(0, min(int(left["end"]), int(right["end"])) - max(int(left["start"]), int(right["start"])))
    union = max(int(left["end"]), int(right["end"])) - min(int(left["start"]), int(right["start"]))
    return intersection / union if union else 0.0


def score_span_labels_detail(model_output: str, task: dict) -> ScoreDetail:
    raw_expected = _config_value(task, "expected_spans", task.get("expected_answer"))
    allowed = {
        _normalize_classification_label(value)
        for value in (_config_value(task, "allowed_labels", []) or [])
    }
    if not isinstance(raw_expected, list) or not allowed:
        return ScoreDetail(None, "span_spec_missing", "span_spec_missing")
    expected = [_valid_span(span, allowed) for span in raw_expected]
    if any(span is None for span in expected):
        return ScoreDetail(None, "invalid_expected_spans", "expected_spans_invalid")
    parsed = _extract_json_object(model_output)
    raw_predicted = parsed.get("spans") if isinstance(parsed, dict) else None
    if not isinstance(raw_predicted, list):
        raw_predicted = []
    predicted = [_valid_span(span, allowed) for span in raw_predicted]
    invalid_count = sum(span is None for span in predicted)
    valid_predicted = [span for span in predicted if span is not None]
    expected_spans = [span for span in expected if span is not None]
    threshold = float(_config_value(task, "span_iou_threshold", 0.5))

    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected_span in enumerate(expected_spans):
        for predicted_index, predicted_span in enumerate(valid_predicted):
            if expected_span["label"] != predicted_span["label"]:
                continue
            overlap = _span_iou(expected_span, predicted_span)
            if overlap >= threshold:
                candidates.append((overlap, expected_index, predicted_index))
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    for _, expected_index, predicted_index in sorted(candidates, reverse=True):
        if expected_index not in matched_expected and predicted_index not in matched_predicted:
            matched_expected.add(expected_index)
            matched_predicted.add(predicted_index)
    true_positive = len(matched_expected)
    false_positive = len(valid_predicted) - true_positive + invalid_count
    false_negative = len(expected_spans) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 1.0
    pass_threshold = float(_config_value(task, "f1_threshold", 1.0))
    passed = f1 >= pass_threshold and invalid_count == 0
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"span_labels_{'passed' if passed else 'failed'} (f1={f1:.3f})",
        "span_labels_passed" if passed else "span_labels_failed",
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "invalid_span_count": invalid_count,
            "f1": round(f1, 6),
            "iou_threshold": threshold,
        },
    )


def score_span_labels(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_span_labels_detail(model_output, task).as_tuple()


def _edit_tokens(value: object) -> list[str]:
    return re.findall(r"[^\W_]+(?:[’'][^\W_]+)*|[^\w\s]", unicodedata.normalize("NFKC", str(value)), re.UNICODE)


def _extract_edits(source: str, target: str) -> set[tuple[int, int, tuple[str, ...]]]:
    source_tokens = _edit_tokens(source)
    target_tokens = _edit_tokens(target)
    edits: set[tuple[int, int, tuple[str, ...]]] = set()
    for operation, source_start, source_end, target_start, target_end in SequenceMatcher(
        None, source_tokens, target_tokens, autojunk=False
    ).get_opcodes():
        if operation != "equal":
            replacement = tuple(token.casefold() for token in target_tokens[target_start:target_end])
            edits.add((source_start, source_end, replacement))
    return edits


def score_copy_edit_f05_detail(model_output: str, task: dict) -> ScoreDetail:
    source = _config_value(task, "source_text")
    expected = task.get("expected_answer")
    if not isinstance(source, str) or not isinstance(expected, str):
        return ScoreDetail(None, "copy_edit_spec_missing", "copy_edit_spec_missing")
    # Existing plain-text contracts stay usable; structured outputs must be a
    # single text field rather than scoring the JSON wrapper as editorial edits.
    if model_output.lstrip().startswith(("{", "```")):
        parsed = _strict_json_object(model_output)
        if parsed is None or set(parsed) != {"text"} or not isinstance(parsed["text"], str):
            return ScoreDetail(0.0, "expected_single_json_text", "format_error")
        prediction = parsed["text"]
    else:
        prediction = _extract_qa_answer(model_output)
    gold_edits = _extract_edits(source, expected)
    predicted_edits = _extract_edits(source, prediction)
    true_positive = len(gold_edits & predicted_edits)
    false_positive = len(predicted_edits - gold_edits)
    false_negative = len(gold_edits - predicted_edits)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else float(not gold_edits)
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    beta_sq = 0.25
    f0_5 = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if precision or recall else 0.0
    )
    protected_values = [
        str(value) for value in (_config_value(task, "protected_values", []) or [])
        if str(value).strip()
    ]
    normalized_prediction = _normalized_text(prediction)
    missing_values = [value for value in protected_values if _normalized_text(value) not in normalized_prediction]
    threshold = float(_config_value(task, "f0_5_threshold", 1.0))
    passed = f0_5 >= threshold and not missing_values
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"copy_edit_{'passed' if passed else 'failed'} (f0.5={f0_5:.3f})",
        "copy_edit_passed" if passed else "copy_edit_failed",
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f0_5": round(f0_5, 6),
            "protected_values_ok": not missing_values,
            "missing_protected_values": missing_values,
        },
    )


def score_copy_edit_f05(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_copy_edit_f05_detail(model_output, task).as_tuple()


def score_atomic_claims_detail(model_output: str, task: dict) -> ScoreDetail:
    raw_catalog = _config_value(task, "claim_catalog", [])
    if not isinstance(raw_catalog, list) or not raw_catalog:
        return ScoreDetail(None, "claim_catalog_missing", "claim_catalog_missing")
    normalized_output = _normalized_text(model_output)
    required_ids: list[str] = []
    matched_required_ids: list[str] = []
    forbidden_ids: list[str] = []
    malformed_ids: list[str] = []
    for index, raw_claim in enumerate(raw_catalog):
        if not isinstance(raw_claim, dict):
            malformed_ids.append(f"claim-{index}")
            continue
        claim_id = str(raw_claim.get("id") or f"claim-{index}")
        status = str(raw_claim.get("status") or "required").casefold()
        alternatives = raw_claim.get("alternatives") or [raw_claim.get("text", "")]
        alternatives = [str(value) for value in alternatives if str(value).strip()]
        if not alternatives or status not in {"required", "supported", "optional", "forbidden", "contradicted", "unsupported"}:
            malformed_ids.append(claim_id)
            continue
        present = any(_normalized_text(value) in normalized_output for value in alternatives)
        if status in {"required", "supported"}:
            required_ids.append(claim_id)
            if present:
                matched_required_ids.append(claim_id)
        elif status in {"forbidden", "contradicted", "unsupported"} and present:
            forbidden_ids.append(claim_id)
    if malformed_ids:
        return ScoreDetail(
            None,
            "claim_catalog_invalid",
            "claim_catalog_invalid",
            {"malformed_claim_ids": malformed_ids},
        )
    coverage = len(matched_required_ids) / len(required_ids) if required_ids else 1.0
    threshold = float(_config_value(task, "min_required_coverage", 1.0))
    passed = coverage >= threshold and not forbidden_ids
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"atomic_claims_{'passed' if passed else 'failed'} (coverage={coverage:.3f})",
        "atomic_claims_passed" if passed else "atomic_claims_failed",
        {
            "required_claim_coverage": round(coverage, 6),
            "matched_required_claim_ids": matched_required_ids,
            "missing_required_claim_ids": [claim_id for claim_id in required_ids if claim_id not in matched_required_ids],
            "forbidden_claim_ids": forbidden_ids,
            "unsupported_claim_rate": len(forbidden_ids) / max(1, len(raw_catalog)),
        },
    )


def score_atomic_claims(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_atomic_claims_detail(model_output, task).as_tuple()


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def _editorial_tokens(value: object) -> list[str]:
    """Tokenize Unicode prose without assuming a language or writing system."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[^\W_]+(?:[’'-][^\W_]+)*", text, flags=re.UNICODE)


def _script_ratio(value: str, script_name: str) -> float:
    visible_letters = [char for char in value if char.isalpha()]
    if not visible_letters:
        return 0.0
    target = script_name.strip().upper()
    matched = sum(target in unicodedata.name(char, "") for char in visible_letters)
    return matched / len(visible_letters)


def score_editorial_rubric_detail(model_output: str, task: dict) -> ScoreDetail:
    """Deterministic, language-neutral editorial gate for sourced generation.

    This is intentionally a gate rather than an LLM judge: it measures reference
    coverage, required facts, an optional configured script ratio, length,
    prohibited language, and citations. Human review remains the final authority.
    """
    expected = str(task.get("expected_answer") or "").strip()
    if not expected:
        return ScoreDetail(None, "no_expected_answer", "expected_answer_missing")

    output_tokens = _editorial_tokens(model_output)
    expected_tokens = _editorial_tokens(expected)
    if not output_tokens:
        return ScoreDetail(0.0, "no_editorial_content", "editorial_content_missing")

    output_set = set(output_tokens)
    expected_set = set(expected_tokens)
    reference_recall = len(output_set & expected_set) / max(1, len(expected_set))

    cfg = task.get("scoring_config") if isinstance(task.get("scoring_config"), dict) else {}
    raw_facts = task.get("required_facts") or cfg.get("required_facts") or []
    required_facts = [str(value) for value in raw_facts if str(value).strip()]
    normalized_output = _normalized_text(model_output)
    matched_facts = sum(
        1 for fact in required_facts
        if _normalized_text(fact) in normalized_output
    )
    fact_coverage = matched_facts / len(required_facts) if required_facts else reference_recall

    raw_forbidden = task.get("forbidden_phrases") or cfg.get("forbidden_phrases") or []
    forbidden = [str(value) for value in raw_forbidden if str(value).strip()]
    forbidden_hits = [phrase for phrase in forbidden if _normalized_text(phrase) in normalized_output]
    if forbidden_hits:
        return ScoreDetail(0.0, "forbidden_editorial_language", "forbidden_language", {"hits": forbidden_hits})

    required_citations = task.get("valid_citations") or cfg.get("valid_citations") or []
    citation_score = 1.0
    if required_citations:
        citation_score = 1.0 if _citation_detail(model_output, task)[0] else 0.0

    min_words = int(task.get("min_words") or cfg.get("min_words") or 1)
    max_words = int(task.get("max_words") or cfg.get("max_words") or 10_000)
    length_score = 1.0 if min_words <= len(output_tokens) <= max_words else 0.0
    required_script = task.get("required_script") or cfg.get("required_script")
    legacy_min_arabic = task.get("min_arabic_ratio") or cfg.get("min_arabic_ratio")
    if not required_script and legacy_min_arabic is not None:
        required_script = "ARABIC"
    min_script = float(task.get("min_script_ratio") or cfg.get("min_script_ratio") or legacy_min_arabic or 0.0)
    script_ratio = _script_ratio(model_output, str(required_script)) if required_script else None
    script_score = 1.0 if script_ratio is None or script_ratio >= min_script else 0.0

    weights = task.get("rubric_weights") or cfg.get("rubric_weights") or {}
    fact_weight = float(weights.get("facts", 0.45))
    reference_weight = float(weights.get("reference", 0.25))
    length_weight = float(weights.get("length", 0.10))
    script_weight = float(weights.get("script", 0.10))
    citation_weight = float(weights.get("citation", 0.10))
    weight_total = fact_weight + reference_weight + length_weight + script_weight + citation_weight
    composite = (
        fact_coverage * fact_weight
        + reference_recall * reference_weight
        + length_score * length_weight
        + script_score * script_weight
        + citation_score * citation_weight
    ) / max(weight_total, 1e-9)
    threshold = float(task.get("pass_threshold") or cfg.get("pass_threshold") or 0.70)
    passed = composite >= threshold and citation_score == 1.0 and script_score == 1.0
    evidence = {
        "composite": round(composite, 4),
        "fact_coverage": round(fact_coverage, 4),
        "reference_recall": round(reference_recall, 4),
        "length_ok": bool(length_score),
        "required_script": required_script,
        "script_ratio": round(script_ratio, 4) if script_ratio is not None else None,
        "citation_ok": bool(citation_score),
    }
    return ScoreDetail(
        1.0 if passed else 0.0,
        f"editorial_rubric_{'passed' if passed else 'failed'} ({composite:.2f})",
        "editorial_rubric_passed" if passed else "editorial_rubric_failed",
        evidence,
    )


def score_editorial_rubric(model_output: str, task: dict) -> tuple[float | None, str]:
    return score_editorial_rubric_detail(model_output, task).as_tuple()
