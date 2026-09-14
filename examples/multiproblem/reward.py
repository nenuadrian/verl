# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward for multi-problem GSM8K prompts.

A response must answer K problems in order, each ending in ``#### <answer>``.
Reward is the fraction of the K that are right, which keeps the signal dense as
K grows -- an all-or-nothing reward would go nearly silent at K=8 and starve
GRPO of within-group variance.

Every call returns the *same* key set regardless of K.  verl zips the per-row
extra-info dicts into flat lists before aggregating, so a K-dependent key set
would silently misalign rows across data sources.  The position keys
(``acc_first``/``acc_last``/``acc_first_half``/``acc_second_half``) are the
headline diagnostic: if packing problems together helps because later problems
can attend to earlier ones, it should show up as a late-position advantage that
grows with K.
"""

import json
import re

# Not anchored to the tail of the string: unlike single-answer GSM8K scoring, we
# need every marker spread through a long multi-solution response.
ANSWER_RE = re.compile(r"####\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def _normalize(text):
    """Canonicalize a numeric answer so '1,200', '$1200' and '1200.0' all match."""
    cleaned = str(text).strip().replace(",", "").replace("$", "").rstrip(".")
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    # Render integral floats without the trailing .0 so string compare suffices.
    return str(int(value)) if value == int(value) else str(value)


def _ground_truths(ground_truth):
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            return [ground_truth]  # plain single-answer row
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(ground_truth, (list, tuple)):
        return list(ground_truth)
    return [ground_truth]


def extract_answers(solution_str):
    """Every ``#### <number>`` in the response, in order."""
    return ANSWER_RE.findall(solution_str or "")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_weight=0.0):
    """Score one multi-problem rollout.

    Args:
        data_source: unused; kept for verl's reward-manager call signature.
        solution_str: decoded response text.
        ground_truth: JSON list of K answer strings (a bare string also works).
        extra_info: unused; K is taken from ground_truth so the two cannot drift.
        format_weight: optional weight on emitting exactly K answers. Default 0
            keeps the reward pure accuracy.

    Returns:
        dict with ``score`` plus fixed-key diagnostics; verl logs every other key
        as ``val-aux/{data_source}/{key}``.
    """
    truths = _ground_truths(ground_truth)
    k = len(truths)
    predictions = extract_answers(solution_str)

    correct = [
        1.0 if i < len(predictions) and _normalize(predictions[i]) == _normalize(truths[i]) else 0.0
        for i in range(k)
    ]
    accuracy = sum(correct) / k
    exact_count = 1.0 if len(predictions) == k else 0.0
    score = (1.0 - format_weight) * accuracy + format_weight * exact_count

    half = max(1, k // 2)
    return {
        "score": score,
        "acc": accuracy,
        "all_correct": 1.0 if accuracy == 1.0 else 0.0,
        "fmt_exact_count": exact_count,
        "n_answers": float(len(predictions)),
        "k": float(k),
        "acc_first": correct[0],
        "acc_last": correct[-1],
        "acc_first_half": sum(correct[:half]) / half,
        "acc_second_half": sum(correct[-half:]) / half,
    }
