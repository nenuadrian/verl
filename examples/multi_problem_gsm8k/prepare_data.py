#!/usr/bin/env python3
"""Build matched multi-problem GSM8K datasets for GRPO experiments.

Every output dataset contains the same shuffled source problems. Only the
boundaries between prompts change with ``K``. Consequently, one epoch of a
K=1 run and one epoch of a K=8 run expose the model to the same number of
underlying GSM8K problems, which is the important control for this experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import datasets


DEFAULT_K_VALUES = (1, 2, 4, 8)
DATA_SOURCE_PREFIX = "multi_problem_gsm8k"
ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def extract_answer(solution: str) -> str:
    """Return the canonical GSM8K answer from a source solution."""
    matches = ANSWER_PATTERN.findall(solution)
    if not matches:
        raise ValueError(f"Could not find a GSM8K final answer in: {solution!r}")
    return matches[-1].replace(",", "")


def make_prompt(questions: Sequence[str]) -> str:
    """Render a single causal-LM prompt containing independent problems."""
    problem_blocks = "\n\n".join(
        f"### Problem {number}\n{question}" for number, question in enumerate(questions, start=1)
    )
    return (
        f"Solve the {len(questions)} independent elementary-school math problems below. "
        "Reason about every problem separately and answer them in order. "
        "End the solution to each problem with exactly one final-answer marker in the form "
        "`#### <number>`. There must be one marker per problem.\n\n"
        f"{problem_blocks}"
    )


def _common_problem_count(num_examples: int, k_values: Sequence[int]) -> int:
    """Return the largest source-problem budget divisible by every requested K."""
    largest_k = max(k_values)
    return num_examples - (num_examples % largest_k)


def build_rows(
    examples: Sequence[Mapping[str, Any]],
    *,
    k: int,
    split: str,
    source_positions: Sequence[int],
    original_source_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Group a fixed, ordered source-problem budget into K-problem prompts."""
    if len(source_positions) % k:
        raise ValueError(f"{len(source_positions)=} must be divisible by {k=}")

    rows: list[dict[str, Any]] = []
    for group_index, offset in enumerate(range(0, len(source_positions), k)):
        selected_positions = source_positions[offset : offset + k]
        group = [examples[index] for index in selected_positions]
        answers = [extract_answer(str(example["answer"])) for example in group]
        questions = [str(example["question"]) for example in group]

        rows.append(
            {
                "data_source": f"{DATA_SOURCE_PREFIX}_k{k}",
                "prompt": [{"role": "user", "content": make_prompt(questions)}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"answers": answers}, separators=(",", ":")),
                },
                "extra_info": {
                    "split": split,
                    "group_index": group_index,
                    "num_problems": k,
                    "source_indices": [original_source_indices[index] for index in selected_positions],
                },
            }
        )
    return rows


def parse_k_values(values: Iterable[str]) -> tuple[int, ...]:
    """Parse and validate the requested K values while preserving their order."""
    k_values = tuple(int(value) for value in values)
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values must contain one or more positive integers")
    if len(set(k_values)) != len(k_values):
        raise ValueError("--k-values must not contain duplicates")
    return k_values


def build_split(
    split_dataset: datasets.Dataset,
    *,
    split: str,
    k_values: Sequence[int],
    seed: int,
    max_problems: int | None,
) -> tuple[dict[int, list[dict[str, Any]]], list[int]]:
    """Choose a shared shuffled budget, then create the rows for each K."""
    available = len(split_dataset)
    if max_problems is not None:
        if max_problems <= 0:
            raise ValueError("--max-*-problems must be positive")
        available = min(available, max_problems)

    common_count = _common_problem_count(available, k_values)
    if common_count == 0:
        raise ValueError(
            f"The {split} split has too few examples ({available}) for max K={max(k_values)}."
        )

    # A split-specific seed leaves train and validation independent while ensuring
    # that every K observes exactly the same source-problem permutation.
    shuffle_seed = seed + (0 if split == "train" else 1)
    source_indices = list(range(len(split_dataset)))
    random.Random(shuffle_seed).shuffle(source_indices)
    source_indices = source_indices[:common_count]

    examples = [split_dataset[index] for index in source_indices]
    local_indices = list(range(common_count))
    rows_by_k = {
        k: build_rows(
            examples,
            k=k,
            split=split,
            source_positions=local_indices,
            original_source_indices=source_indices,
        )
        for k in k_values
    }
    # Record original GSM8K row ids in the manifest, but keep parquet fields
    # compact and directly useful for debugging generated samples.
    return rows_by_k, source_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-save-dir",
        type=Path,
        required=True,
        help="Directory that will receive parquets and manifest.json.",
    )
    parser.add_argument(
        "--local-dataset-path",
        default=None,
        help="Optional local GSM8K dataset path. Defaults to openai/gsm8k (main).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=[str(k) for k in DEFAULT_K_VALUES],
        help="Prompt sizes to generate. Defaults to 1 2 4 8.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-problems",
        type=int,
        default=None,
        help="Optional cap on source train problems; useful for smoke tests.",
    )
    parser.add_argument(
        "--max-test-problems",
        type=int,
        default=None,
        help="Optional cap on source test problems; useful for smoke tests.",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    save_dir = args.local_save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(args.local_dataset_path or "openai/gsm8k", "main")
    train_rows, train_indices = build_split(
        dataset["train"],
        split="train",
        k_values=k_values,
        seed=args.seed,
        max_problems=args.max_train_problems,
    )
    val_rows, val_indices = build_split(
        dataset["test"],
        split="test",
        k_values=k_values,
        seed=args.seed,
        max_problems=args.max_test_problems,
    )

    for k, rows in train_rows.items():
        datasets.Dataset.from_list(rows).to_parquet(str(save_dir / f"train_k{k}.parquet"))
        print(f"train_k{k}.parquet: {len(rows)} prompts / {len(rows) * k} source problems")
    for k, rows in val_rows.items():
        datasets.Dataset.from_list(rows).to_parquet(str(save_dir / f"val_k{k}.parquet"))
        print(f"val_k{k}.parquet: {len(rows)} prompts / {len(rows) * k} source problems")

    combined_val_rows = [row for k in k_values for row in val_rows[k]]
    datasets.Dataset.from_list(combined_val_rows).to_parquet(str(save_dir / "val_all.parquet"))
    print(f"val_all.parquet: {len(combined_val_rows)} prompts across K={list(k_values)}")

    manifest = {
        "source_dataset": args.local_dataset_path or "openai/gsm8k:main",
        "seed": args.seed,
        "k_values": list(k_values),
        "train_source_problem_count": len(train_indices),
        "test_source_problem_count": len(val_indices),
        "train_source_indices": train_indices,
        "test_source_indices": val_indices,
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
