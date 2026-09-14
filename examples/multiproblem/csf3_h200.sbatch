#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Multi-problem GSM8K GRPO for Qwen2.5-3B-Instruct on CSF3 H200.
#
#   sbatch --job-name=mp-k4 examples/multiproblem/csf3_h200.sbatch 4 [hydra overrides...]
#
# The gpu-h200-fse QOS caps a user at 4 h200 GPUs, so the K sweep runs as four
# concurrent single-GPU jobs rather than one multi-GPU job.
#
# Python stack: the conda `verl` env supplies the interpreter, ray, hydra and
# omegaconf, while ~/.local/lib/python3.12 supplies torch 2.11 / vLLM 0.24 /
# transformers 5.12, which is the combination verl 0.10 requires.  So, unlike
# the older 0.7-era scripts in this cluster, PYTHONNOUSERSITE is deliberately
# NOT set here -- user site-packages are the point.  Nothing in either tree is
# modified; the few missing pure-python deps live in a private .deps dir.
set -euo pipefail

K="${1:?usage: sbatch csf3_h200.sbatch <K> [hydra overrides...]}"
shift || true

module purge
module load miniforge
conda activate verl

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ENV_DIR="${HOME}/.conda/envs/verl"
cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/slurm_logs"

export PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}/.deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# verl rejects ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES, and vLLM's
# CuMem pool is incompatible with expandable_segments:True.
unset ROCR_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF

export HF_HOME=/scratch/mbax2an2/hf_cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MODEL_PATH="${MODEL_PATH:-/scratch/mbax2an2/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
export DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/gsm8k_multiproblem}"

export TMPDIR="/tmp/mp_gsm8k_${SLURM_JOB_ID:-local}"
export RAY_TMPDIR="${TMPDIR}/ray"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${TRITON_CACHE_DIR}"
export RAY_memory_usage_threshold=0.9
export RAY_memory_monitor_refresh_ms=1000

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export PROJECT_NAME="${PROJECT_NAME:-verl_gsm8k_multiproblem_qwen25_3b}"
export BUDGET_MODE="${BUDGET_MODE:-problems}"
export PROBLEMS_PER_STEP="${PROBLEMS_PER_STEP:-256}"
export EXP_NAME="${EXP_NAME:-k${K}_${BUDGET_MODE}${PROBLEMS_PER_STEP}_h200_${SLURM_JOB_ID:-local}}"

export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO_DIR}/wandb"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-multiproblem_ksweep}"
export WANDB_TAGS="multiproblem,gsm8k,qwen25-3b,grpo,k${K},${BUDGET_MODE},h200"
mkdir -p "${WANDB_DIR}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"${PYTHON_BIN}" -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

test -s "${MODEL_PATH}/config.json"
test -s "${DATA_DIR}/train_k${K}.parquet"

exec env K="${K}" bash examples/multiproblem/run_gsm8k_multiprob.sh "$@"
