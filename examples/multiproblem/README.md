# Multi-problem prompts: does packing K problems into one context change what RL teaches the model?

GRPO on GSM8K with Qwen2.5-3B-Instruct, where each prompt contains **K independently
sampled problems** that a single rollout must solve in one attention context.

The literature that exists (multi-problem prompting / multi-problem evaluation) asks
whether an *already-trained* model can answer several questions in one prompt. This
recipe asks a different question: if you **train** with K problems per sequence, does
the model get better per problem consumed, and does it get better *at later positions*
in a way that implicates cross-problem attention?

## The design

| Regime | Prompt | Question |
| --- | --- | --- |
| `K=1` | one problem | ordinary GRPO baseline |
| `K>1` | K problems, one rollout | does packing help per problem consumed? |
| eval at every K | all runs scored on K∈{1,2,4,8} | does training K transfer to other K? |

### The control that makes it a comparison

`prepare_data.py` cuts every K from **one fixed shuffle of one problem pool**:
K=1 gets `[p0]`, `[p1]`, …; K=2 gets `[p0,p1]`, `[p2,p3]`, …; K=8 gets `[p0..p7]`, ….
So the shards are nested and every K covers **exactly the same 7472 train problems in
the same order** — only the grouping differs. The pool is trimmed to a multiple of
`lcm(K)` so no K drops a partial tail.

The launcher then sets `train_batch_size = PROBLEMS_PER_STEP / K`, so each step
consumes the same number of *problems* at every K, and an epoch is the same number of
optimizer steps at every K. That makes "problems seen" an exactly matched budget —
the resource the comparison is actually about. `BUDGET_MODE=prompts` holds the prompt
count fixed instead (K× more problems per step at larger K) if you want the other axis.

### What is deliberately *not* matched

Two things cannot be held fixed at the same time as the problem budget, and both are
inherent to the design rather than fixable:

- **Sequence length grows with K.** K=8 generates ~4× the tokens of K=2 per step, so
  wall-clock per step is not comparable even though the problem budget is.
- **GRPO group count shrinks with K** under `problems` mode (fewer prompts/step, same
  group size `rollout.n`), so advantage estimation sees fewer groups per update.

Report both, don't hide them.

What *is* matched, deliberately: the **generation budget per problem**. Response length
is `RESPONSE_TOKENS_PER_PROBLEM * K` (384 by default), not a constant plus a slope — a
fixed additive term would hand low-K runs a larger per-problem token budget and bias
the sweep against the packed conditions. Prompt limits are sized from the measured
Qwen2.5 maxima over the built shards (288 / 350 / 540 / 854 tokens at K = 1 / 2 / 4 / 8),
so nothing is ever truncated.

### The ablation that actually attributes the effect

If K>1 beats K=1, that alone does **not** show cross-problem attention is doing the
work — a different data ordering and loss structure could explain it. The decisive
follow-up is to re-run K>1 with attention **masked at problem boundaries** (block-
diagonal within the prompt, so problem *j* cannot see problem *i<j*). If the advantage
survives masking, it was never about within-context computation. That ablation is not
in this recipe yet; it needs a document-boundary mask threaded into the actor forward
pass, and it is the natural phase 2.

## Prompt and reward

Every K uses a byte-identical template — no problem count appears in the wording, so
K=1 is not a different prompt distribution:

```
Problem 1:
<question>

Problem 2:
<question>

Solve every problem above. Reason step by step for each one, and end each problem's
solution with its final answer on its own line in the form "#### <answer>". Give the
solutions in problem order.
```

`reward.py` extracts every `#### <number>` in order and scores position *j* against
ground truth *j*. Reward is the **fraction of K correct** — dense on purpose, since
all-or-nothing would go nearly silent at K=8 and starve GRPO of within-group variance.

Every call returns the same key set regardless of K (verl zips per-row extra-info into
flat lists before aggregating, so a K-dependent key set would silently misalign rows).

## Reading the results in W&B

Each K is tagged as its own `data_source` (`gsm8k_mp_k{K}`), and all runs validate on
**all** of them, so one run reports a full row of the train-K × eval-K transfer matrix:

| metric | meaning |
| --- | --- |
| `val-core/gsm8k_mp_k{K}/reward/mean@1` | per-problem accuracy at eval K |
| `val-aux/gsm8k_mp_k{K}/acc_first/mean@1` | accuracy on problem 1 |
| `val-aux/gsm8k_mp_k{K}/acc_last/mean@1` | accuracy on problem K |
| `val-aux/gsm8k_mp_k{K}/acc_first_half`, `acc_second_half` | the position effect |
| `val-aux/gsm8k_mp_k{K}/all_correct/mean@1` | strict all-K-correct rate |
| `val-aux/gsm8k_mp_k{K}/fmt_exact_count/mean@1` | emitted exactly K answers |

**The headline signal is `acc_last` − `acc_first` as a function of K.** If packing helps
because later problems can attend to earlier ones, it shows up as a late-position
advantage that grows with K. If instead accuracy *decays* with position, that is
context interference, which is the more likely null result and worth reporting plainly.

`val-core/gsm8k_mp_k1/...` is ordinary single-problem GSM8K accuracy, so every run is
always comparable to a standard baseline.

## Running it

```bash
python3 examples/multiproblem/prepare_data.py \
  --local_save_dir data/gsm8k_multiproblem --k 1 2 4 8 --val_problems 512 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct           # --tokenizer prints length percentiles

K=4 bash examples/multiproblem/run_gsm8k_multiprob.sh                 # one K, locally
bash examples/multiproblem/submit_sweep.sh                            # K=1,2,4,8 on CSF3
KS="1 8" TIME=01:00:00 bash examples/multiproblem/submit_sweep.sh trainer.total_training_steps=5
```

`--from_verl_parquet <dir>` rebuilds from existing verl-format GSM8K parquets instead
of hitting the Hub, which is what you want on a cluster with no compute-node network.

`data.truncation=error` is set deliberately: an undersized `max_prompt_length` fails
loudly rather than silently dropping problems and breaking the matched budget. Size the
limits from the percentiles `--tokenizer` prints.

## CSF3 notes

- `gpu-h200-fse` QOS caps a **user at 4 h200 GPUs**, so the sweep runs as four
  concurrent single-GPU jobs rather than one multi-GPU job.
- verl 0.10 needs torch 2.11 / vLLM 0.24 / transformers ≥5.5. On CSF3 that combination
  lives in `~/.local/lib/python3.12` (user site-packages), while the conda `verl` env
  supplies the interpreter, ray, hydra and omegaconf. So `csf3_h200.sbatch`
  deliberately does **not** set `PYTHONNOUSERSITE` — unlike the older 0.7-era scripts
  on this cluster, user site-packages are the point. Missing pure-python deps go in a
  private `.deps/` dir so neither tree is modified.
- `use_remove_padding` defaults to **False**: verl's rmpad path imports `flash_attn`
  unguarded, and the installed `flash_attn` was built against a different torch. The
  cost is throughput only and it is identical at every K.
