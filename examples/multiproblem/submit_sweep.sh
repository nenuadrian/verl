#!/usr/bin/env bash
# Submit the K sweep on CSF3: one single-GPU H200 job per K.
#
#   bash examples/multiproblem/submit_sweep.sh                  # K = 1 2 4 8
#   KS="1 8" TIME=02:00:00 bash examples/multiproblem/submit_sweep.sh
#   bash examples/multiproblem/submit_sweep.sh trainer.total_training_steps=5   # extra hydra overrides
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p slurm_logs

KS="${KS:-1 2 4 8}"
TIME="${TIME:-06:00:00}"
PREFIX="${PREFIX:-mp}"

# The gpu-h200-fse QOS allows 4 GPUs per user in total; anything beyond that
# queues rather than failing, but it is worth saying so up front.
n_jobs=$(wc -w <<< "${KS}")
echo "submitting ${n_jobs} single-GPU jobs (QOS cap is 4 concurrent h200 GPUs)"

for k in ${KS}; do
  jid=$(sbatch --parsable \
    --job-name="${PREFIX}-k${k}" \
    --time="${TIME}" \
    examples/multiproblem/csf3_h200.sbatch "${k}" "$@")
  echo "K=${k}  job ${jid}  logs: slurm_logs/${PREFIX}-k${k}_${jid}.{out,err}"
done

echo
squeue -u "${USER}" -o "%.12i %.14j %.2t %.10M %.6D %R"
