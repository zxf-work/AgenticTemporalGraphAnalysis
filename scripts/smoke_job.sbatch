#!/bin/bash
# Short shakedown job: start vLLM for one checkpoint, verify tool calling, run 2 real episodes.
# Site-specific: set your own account/partition/gres. The ladder was served on one
# H100 80GB per checkpoint.
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=atgp-smoke
#SBATCH --output=logs/smoke-%j.out
set -uo pipefail
SLOT="${1:-qwen7b}"
cd "$HOME/temporal-agent-probes"; mkdir -p logs runs
source $PROJECT/venvs/atgp2/bin/activate
export HF_HOME=$PROJECT/hf VLLM_LOGGING_LEVEL=WARNING
MODEL_ID=$(python -c "import yaml;print(yaml.safe_load(open('configs/models.yaml'))['models']['$SLOT']['id'])")
PORT=$((8000 + (SLURM_JOB_ID % 1000)))
echo "== slot=$SLOT model=$MODEL_ID node=$(hostname) port=$PORT"
vllm serve "$MODEL_ID" --port "$PORT" --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests \
  > "logs/vllm-smoke-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "== healthy after ${i}0s"; break; }
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "!! vLLM died"; tail -40 "logs/vllm-smoke-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "!! never healthy"; tail -40 "logs/vllm-smoke-$SLURM_JOB_ID.log"; kill $VLLM_PID; exit 1; }
export ATGP_VLLM_URL="http://127.0.0.1:$PORT/v1" ATGP_VLLM_KEY=EMPTY
python scripts/smoke_vllm.py "$SLOT"; rc=$?
if [ $rc -eq 0 ]; then
  echo "== 2 real episodes (1 seed, R=32, A1)"
  python scripts/run_matrix.py ladder_m1_a1 --conditions A1 --models "$SLOT" \
      --out "runs/smoke_$SLOT" --seeds 1 --values 32 --conc 2 2>&1 | tail -14 || rc=$?
fi
kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
echo "== smoke done rc=$rc"; exit $rc
