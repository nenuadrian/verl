#!/usr/bin/env bash
# GRPO on multi-problem GSM8K for one value of K.
#
#   K=4 bash examples/multiproblem/run_gsm8k_multiprob.sh [hydra overrides...]
#
# Budget control: the per-step *problem* budget is held fixed across the sweep by
# setting train_batch_size = PROBLEMS_PER_STEP / K.  Combined with the shared
# problem pool built by prepare_data.py, every K therefore consumes exactly the
# same GSM8K problems in the same order at the same rate, and runs the same
# number of optimizer steps per epoch.  Set BUDGET_MODE=prompts to hold the
# prompt count fixed instead (K x more problems per step at larger K).
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
unset ROCR_VISIBLE_DEVICES

k="${K:?set K to the number of problems per prompt}"
data_dir="${DATA_DIR:-${ROOT_DIR}/data/gsm8k_multiproblem}"
model_path="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

train_file="${data_dir}/train_k${k}.parquet"
[[ -f "${train_file}" ]] || { echo "missing ${train_file}; run prepare_data.py first" >&2; exit 1; }

# Validate on every K, not just the training K.  verl buckets val metrics by
# data_source, so one run reports the whole train-K x eval-K transfer row.
eval_ks="${EVAL_KS:-1 2 4 8}"
val_list=""
for ek in ${eval_ks}; do
  vf="${data_dir}/test_k${ek}.parquet"
  [[ -f "${vf}" ]] || { echo "missing ${vf}" >&2; exit 1; }
  val_list="${val_list:+${val_list},}'${vf}'"
done

budget_mode="${BUDGET_MODE:-problems}"
problems_per_step="${PROBLEMS_PER_STEP:-256}"
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  train_batch_size="${TRAIN_BATCH_SIZE}"
elif [[ "${budget_mode}" == "problems" ]]; then
  (( problems_per_step % k == 0 )) || { echo "PROBLEMS_PER_STEP must be divisible by K" >&2; exit 1; }
  train_batch_size=$(( problems_per_step / k ))
else
  train_batch_size="${problems_per_step}"
fi
# One optimizer update per fresh rollout batch, at every K.
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-${train_batch_size}}"

# Defaults grow with K; override from the measured percentiles that
# prepare_data.py --tokenizer prints.  truncation=error below makes an
# undersized limit fail loudly rather than silently drop problems and break the
# matched-budget control.
max_prompt_length="${MAX_PROMPT_LENGTH:-$(( 256 + 128 * k ))}"
max_response_length="${MAX_RESPONSE_LENGTH:-$(( 256 + 256 * k ))}"
max_model_len="${MAX_MODEL_LEN:-$(( max_prompt_length + max_response_length ))}"

# Remove-padding is off by default: verl's rmpad path imports flash_attn
# unguarded, and flash_attn is unusable on stacks where it was built against a
# different torch. The cost is throughput only, and it is identical at every K,
# so it cannot confound the sweep. Set USE_REMOVE_PADDING=True where flash_attn
# is healthy.
rollout_n="${ROLLOUT_N:-8}"
actor_lr="${ACTOR_LR:-1e-6}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
use_kl_loss="${USE_KL_LOSS:-True}"
entropy_coeff="${ENTROPY_COEFF:-0.0}"
temperature="${TEMPERATURE:-1.0}"

# Dynamic batching keeps a single micro-batch rule working across a 4x swing in
# sequence length; a fixed micro_batch_size_per_gpu would need retuning per K.
ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
rollout_tp="${ROLLOUT_TP:-1}"
ngpus_per_node="${NGPUS_PER_NODE:-1}"
nnodes="${NNODES:-1}"

project_name="${PROJECT_NAME:-verl_gsm8k_multiproblem_qwen25_3b}"
exp_name="${EXP_NAME:-k${k}_${budget_mode}${problems_per_step}_n${rollout_n}_$(date +%Y%m%d_%H%M%S)}"
out_dir="${OUT_DIR:-${ROOT_DIR}/outputs/${project_name}/${exp_name}}"
mkdir -p "${out_dir}"

logger="${LOGGER:-[\"console\",\"wandb\"]}"


# verl moved the reward hook under `reward.` and dropped `actor_loss_type` between
# 0.7 and 0.10. Detect the layout so this recipe runs unchanged on either.
compat_args=()
if [[ -f "${ROOT_DIR}/verl/trainer/config/reward/reward.yaml" ]]; then
  compat_args+=("reward.custom_reward_function.path=${SCRIPT_DIR}/reward.py"
                "reward.custom_reward_function.name=compute_score")
else
  compat_args+=("custom_reward_function.path=${SCRIPT_DIR}/reward.py"
                "custom_reward_function.name=compute_score"
                "actor_rollout_ref.actor.actor_loss_type=ppo")
fi

echo "============================================================"
echo "[multiproblem] K=${k} budget=${budget_mode}/${problems_per_step}"
echo "train_batch_size=${train_batch_size} (prompts) -> $(( train_batch_size * k )) problems/step"
echo "rollout.n=${rollout_n}  seq=${max_prompt_length}+${max_response_length}"
echo "train=${train_file}"
echo "val=[${val_list}]"
echo "============================================================"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['${train_file}']" \
  data.val_files="[${val_list}]" \
  data.train_batch_size="${train_batch_size}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-256}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.shuffle=True \
  data.validation_shuffle=False \
  data.seed="${DATA_SEED:-42}" \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING:-False}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS:-1}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ppo_max_token_len_per_gpu}" \
  actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}" \
  actor_rollout_ref.actor.entropy_coeff="${entropy_coeff}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME:-vllm}" \
  actor_rollout_ref.rollout.n="${rollout_n}" \
  actor_rollout_ref.rollout.temperature="${temperature}" \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
  actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-256}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  critic.enable=false \
  reward_model.enable=False \
  algorithm.use_kl_in_reward="${USE_KL_IN_REWARD:-False}" \
  trainer.logger="${logger}" \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node="${ngpus_per_node}" \
  trainer.nnodes="${nnodes}" \
  trainer.save_freq="${SAVE_FREQ:--1}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-20}" \
  trainer.default_local_dir="${out_dir}" \
  "${compat_args[@]}" \
  "$@" 2>&1 | tee "${out_dir}/train.log"
