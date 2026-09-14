# Multi-problem GSM8K GRPO sweep

This recipe tests whether training on several independent GSM8K problems in one
causal-LM context changes learning compared with ordinary one-problem prompts.
It runs Qwen2.5-3B-Instruct with GRPO for `K = 1, 2, 4, 8`.

The data-preparation step uses one fixed shuffled source-problem sequence. It
drops only the smallest remainder needed to make every requested `K` divide the
budget, then changes only prompt boundaries. With the default GSM8K splits,
every training condition sees 7,472 source problems per epoch and every
validation condition is evaluated on the same 1,312 source problems. The
launcher also uses `128 / K` prompts per update and a response cap of `512 * K`.
This matches source-problem exposure, prompt-batch problem count, and the
maximum response-token allowance per underlying problem.

`val_all.parquet` contains K=1, 2, 4, and 8 prompts. Its `data_source` is
`multi_problem_gsm8k_k<K>`, so verl reports a train-K-by-eval-K matrix in W&B.
The key validation fields are `answer_accuracy`, `all_correct`,
`format_valid`, and `answer_markers` under each data source.

## Prepare the data

From the repository root:

```bash
python3 examples/multi_problem_gsm8k/prepare_data.py \
  --local-save-dir /path/to/multi-problem-gsm8k/data
```

For a small schema-only smoke test, use `--max-train-problems 32
--max-test-problems 32`. The generated `manifest.json` records the exact source
indices and seed.

## Launch one condition

```bash
K=4 DATA_DIR=/path/to/multi-problem-gsm8k/data \
  bash examples/multi_problem_gsm8k/run_qwen2_5_3b_grpo.sh
```

The script defaults to a one-GPU FSDP + vLLM setup. It accepts normal Hydra
overrides after the command, and environment variables such as `MODEL_PATH`,
`TOTAL_EPOCHS`, `CHECKPOINT_DIR`, `BASE_PROBLEMS_PER_UPDATE`, and `ROLLOUT_N`.

## CSF3 H200 sweep

The Slurm script uses the existing CSF3 W&B authentication. Set up an
H200-scratch checkout and data directory:

```bash
export WORK_ROOT=/mnt/h200-scratch/$USER/multi-problem-gsm8k
git clone --branch <branch> <repository-url> "$WORK_ROOT/verl"
PYTHONPATH="$WORK_ROOT/verl" "$HOME/.conda/envs/verl/bin/python" \
  "$WORK_ROOT/verl/examples/multi_problem_gsm8k/prepare_data.py" \
  --local-save-dir "$WORK_ROOT/data"
cd "$WORK_ROOT/verl"
sbatch --array=0-3 examples/multi_problem_gsm8k/csf3_h200_sweep.sbatch
```

The array uses one H200 per condition, requests the `gpu-h200-fse` QoS, sets a
shared W&B run group, and keeps checkpoints, logs, and cache in H200 scratch.

This is the direct multi-problem comparison. It does not yet mask attention at
problem boundaries; that causal-attention ablation needs a model-side attention
mask change rather than a data or trainer configuration override.
