#!/usr/bin/env bash
# Submit with: sbatch --array=0-3 examples/multi_problem_gsm8k/csf3_h200_sweep.sbatch
# The launcher assigns K=1,2,4,8 to array indices 0,1,2,3 respectively.

#SBATCH --job-name=mp-gsm8k
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/h200-scratch/%u/multi-problem-gsm8k/logs/%x-%A_%a.out
#SBATCH --error=/mnt/h200-scratch/%u/multi-problem-gsm8k/logs/%x-%A_%a.err

set -euo pipefail
set -x

K_VALUES=(1 2 4 8)
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array.}
K=${K_VALUES[${TASK_INDEX}]:-}
if [[ -z "${K}" ]]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_INDEX}; expected 0 through 3." >&2
    exit 2
fi

WORK_ROOT=${WORK_ROOT:-/mnt/h200-scratch/${USER}/multi-problem-gsm8k}
REPO_DIR=${REPO_DIR:-${WORK_ROOT}/verl}
DATA_DIR=${DATA_DIR:-${WORK_ROOT}/data}
# vLLM 0.24 imports jmespath transitively via its SageMaker entrypoint but does
# not declare it, so the rollout replica fails to import. Keep the shim in a
# private dir rather than mutating the shared conda env:
#   python -m pip install --no-deps --target "${WORK_ROOT}/.deps" jmespath
DEPS_DIR=${DEPS_DIR:-${WORK_ROOT}/.deps}
ENV_PYTHON=${ENV_PYTHON:-${HOME}/.conda/envs/verl/bin/python}
PROJECT_NAME=${PROJECT_NAME:-multi-problem-gsm8k}

ENV_PREFIX=$(cd -- "$(dirname -- "${ENV_PYTHON}")/.." && pwd)
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/ray" "${WORK_ROOT}/checkpoints"
export HF_HOME=${HF_HOME:-${WORK_ROOT}/hf-cache}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/${USER}/ray-k${K}}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen2_5_3b-matched-k-sweep}
export WANDB_NAME=${WANDB_NAME:-qwen2_5_3b-grpo-k${K}}

if [[ ! -x "${ENV_PYTHON}" ]]; then
    echo "Python environment not found: ${ENV_PYTHON}" >&2
    exit 2
fi
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "Repository checkout not found: ${REPO_DIR}. Run the documented setup first." >&2
    exit 2
fi
if [[ ! -f "${DATA_DIR}/train_k${K}.parquet" || ! -f "${DATA_DIR}/val_all.parquet" ]]; then
    echo "Prepared multi-problem parquet files are missing in ${DATA_DIR}." >&2
    exit 2
fi

cd "${REPO_DIR}"
"${ENV_PYTHON}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("wandb") is None:
    sys.exit("wandb is not installed in ENV_PYTHON")
PY

"${ENV_PYTHON}" examples/multi_problem_gsm8k/prepare_data.py --help >/dev/null
PATH="$(dirname "${ENV_PYTHON}"):${PATH}" \
PYTHONPATH="${DEPS_DIR}:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
K="${K}" \
DATA_DIR="${DATA_DIR}" \
CHECKPOINT_DIR="${WORK_ROOT}/checkpoints/k${K}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="qwen2_5_3b-grpo-k${K}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}" \
TEST_FREQ="${TEST_FREQ:-10}" \
SAVE_FREQ="${SAVE_FREQ:-29}" \
bash examples/multi_problem_gsm8k/run_qwen2_5_3b_grpo.sh
