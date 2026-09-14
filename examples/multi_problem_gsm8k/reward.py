"""Per-problem exact-match reward for multi-problem GSM8K prompts."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any


ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def _canonical_number(value: str) -> str | None:
    """Normalize common numeric renderings without accepting non-numeric text."""
    candidate = str(value).strip().replace(",", "").replace("$", "")
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    # Decimal normalizes 1.0 and 1 to the same exact value without float rounding.
    return format(number.normalize(), "f")


def _expected_answers(ground_truth: str | dict[str, Any]) -> list[str]:
    """Decode the compact ground-truth payload made by ``prepare_data.py``."""
    payload = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, list) or not answers:
        raise ValueError("Expected ground_truth to be {'answers': [<one or more answers>]}")
    canonical = [_canonical_number(answer) for answer in answers]
    if any(answer is None for answer in canonical):
        raise ValueError(f"Ground truth contains an invalid numeric answer: {answers!r}")
    return canonical  # type: ignore[return-value]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """Score answer markers in order and expose diagnostics to validation logging.

    The scalar GRPO reward is the fraction of independently correct answers. The
    extra fields make W&B report per-problem accuracy, strict all-correct rate,
    formatting compliance, and answer-marker count for every evaluation K.
    """
    if not str(data_source).startswith("multi_problem_gsm8k_k"):
        raise ValueError(f"Unexpected data source for multi-problem reward: {data_source!r}")

    expected = _expected_answers(ground_truth)
    predictions = [_canonical_number(answer) for answer in ANSWER_PATTERN.findall(solution_str)]
    k = len(expected)

    # Align marker i with Problem i. Missing markers are necessarily incorrect;
    # extra markers are ignored for partial credit but fail the strict format metric.
    correct = sum(
        prediction == target
        for prediction, target in zip(predictions[:k], expected)
    )
    format_valid = float(len(predictions) == k and all(prediction is not None for prediction in predictions))
    answer_accuracy = correct / k
    all_correct = float(format_valid and correct == k)

    return {
        "score": answer_accuracy,
        "answer_accuracy": answer_accuracy,
        "all_correct": all_correct,
        "format_valid": format_valid,
        "answer_markers": float(len(predictions)),
    }
