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
"""Build multi-problem GSM8K parquet shards for a K-sweep.

Each row packs K independently sampled GSM8K problems into a single prompt, so
one rollout has to solve all K inside one attention context.

The shards for every K are cut from *one* fixed shuffle of the same problem
pool: K=1 gets [p0], [p1], ...; K=2 gets [p0,p1], [p2,p3], ...; K=8 gets
[p0..p7], ....  So every K covers exactly the same problems in the same order
and only the grouping differs.  That is what makes "problems seen" an
exactly-matched budget across the sweep, which is the control the whole
comparison rests on.

Each K is written under its own ``data_source`` tag (``gsm8k_mp_k{K}``).  verl
buckets validation metrics by data_source, so passing every ``test_k*.parquet``
as val_files makes each run report its accuracy at every K without any extra
plumbing -- i.e. the full train-K x eval-K transfer matrix.
"""

import argparse
import json
import math
import os
import re

import datasets
import pandas as pd

ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

# Deliberately free of any problem count, so the wording is byte-identical for
# every K.  A "solve the following 1 problems" style template would make K=1 a
# different prompt distribution and confound the comparison at its cheapest
# point.
INSTRUCTION = (
    "Solve every problem above. Reason step by step for each one, and end each "
    'problem\'s solution with its final answer on its own line in the form '
    '"#### <answer>". Give the solutions in problem order.'
)


def extract_solution(solution_str):
    solution = ANSWER_RE.search(solution_str)
    assert solution is not None, f"no #### answer in: {solution_str[:200]}"
    return solution.group(1).replace(",", "")


def build_prompt(questions):
    """Render K questions as one numbered multi-problem prompt."""
    blocks = [f"Problem {i}:\n{q.strip()}" for i, q in enumerate(questions, start=1)]
    return "\n\n".join(blocks) + "\n\n" + INSTRUCTION


def load_pool(args, split):
    """Return (questions, answers) for a split, from HF or existing verl parquets."""
    if args.from_verl_parquet:
        path = os.path.join(os.path.expanduser(args.from_verl_parquet), f"{split}.parquet")
        frame = pd.read_parquet(path)
        questions = [str(info["question"]) for info in frame["extra_info"]]
        answers = [str(rm["ground_truth"]) for rm in frame["reward_model"]]
        return questions, answers

    hf_split = "train" if split == "train" else "test"
    dataset = datasets.load_dataset(args.data_source, "main")[hf_split]
    questions = [q.strip() for q in dataset["question"]]
    answers = [extract_solution(a) for a in dataset["answer"]]
    return questions, answers


def build_rows(questions, answers, order, k, split, index_offset=0):
    """Chunk ``order`` into groups of K and render one dataset row per group."""
    rows = []
    for row_idx, start in enumerate(range(0, len(order), k)):
        group = order[start : start + k]
        if len(group) < k:  # only reachable if the pool was not trimmed; skip partials
            break
        group_questions = [questions[i] for i in group]
        group_answers = [answers[i] for i in group]
        rows.append(
            {
                "data_source": f"gsm8k_mp_k{k}",
                "prompt": [{"role": "user", "content": build_prompt(group_questions)}],
                "ability": "math",
                # Stored as JSON rather than a nested list: parquet/numpy object
                # round-tripping of ragged lists through verl's non_tensor_batch
                # is fragile, and a string survives it untouched.
                "reward_model": {"style": "rule", "ground_truth": json.dumps(group_answers)},
                "extra_info": {
                    "split": split,
                    "index": index_offset + row_idx,
                    "k": k,
                    "answers": json.dumps(group_answers),
                    "pool_indices": json.dumps([int(i) for i in group]),
                },
            }
        )
    return rows


def report_token_stats(rows, tokenizer_path, k):
    """Print prompt-length percentiles so max_prompt_length can be set from data."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    lengths = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    lengths.sort()

    def pct(p):
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    print(
        f"  k={k:<2d} prompt tokens: p50={pct(0.50):<5d} p90={pct(0.90):<5d} "
        f"p99={pct(0.99):<5d} max={lengths[-1]:<5d}"
    )
    return lengths[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8], help="problems per prompt")
    parser.add_argument("--local_save_dir", default="~/data/gsm8k_multiproblem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val_problems",
        type=int,
        default=512,
        help="GSM8K test problems to evaluate on; the same ones are used for every K",
    )
    parser.add_argument("--data_source", default="openai/gsm8k")
    parser.add_argument(
        "--from_verl_parquet",
        default=None,
        help="rebuild from an existing verl gsm8k data dir (train.parquet/test.parquet) instead of HF",
    )
    parser.add_argument("--tokenizer", default=None, help="if set, report prompt token percentiles")
    args = parser.parse_args()

    ks = sorted(set(args.k))
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Trim each pool to a multiple of lcm(K) so no K has to drop a partial tail:
    # every K then consumes byte-for-byte the same problems.
    stride = math.lcm(*ks)

    summary = []
    for split in ("train", "test"):
        questions, answers = load_pool(args, split)
        order = list(range(len(questions)))
        rng = __import__("random").Random(args.seed if split == "train" else args.seed + 1)
        rng.shuffle(order)

        if split == "test" and args.val_problems > 0:
            order = order[: args.val_problems]
        usable = (len(order) // stride) * stride
        if usable == 0:
            raise SystemExit(f"{split}: pool of {len(order)} too small for stride {stride}")
        dropped = len(order) - usable
        order = order[:usable]
        print(f"[{split}] pool={len(questions)} used={usable} dropped={dropped} (stride={stride})")

        for k in ks:
            rows = build_rows(questions, answers, order, k, split)
            out = os.path.join(save_dir, f"{split}_k{k}.parquet")
            pd.DataFrame(rows).to_parquet(out)
            print(f"  wrote {out}  rows={len(rows)}  problems={len(rows) * k}")
            if args.tokenizer:
                report_token_stats(rows, args.tokenizer, k)
            summary.append((split, k, len(rows), len(rows) * k))

    print("\nsplit  k   rows    problems")
    for split, k, rows, problems in summary:
        print(f"{split:<6} {k:<3d} {rows:<7d} {problems}")
    print(
        "\nEvery k within a split covers the same problem set; only the grouping differs.\n"
        "Set data.train_batch_size = PROBLEMS_PER_STEP / k to hold the per-step\n"
        "problem budget fixed across the sweep."
    )


if __name__ == "__main__":
    main()
