#!/bin/bash
# Milestone C4 — run the open-weight ladder for ONE checkpoint.
#
#   sbatch scripts/ladder_job.sbatch <model_slot>      e.g. qwen7b | qwen14b | qwen32b
#
# One job = one model = one vLLM server + the four preregistered sweeps. Serving and running in
# the same job keeps the endpoint private to this job (no port collisions between concurrent
# checkpoints) and means the GPU is released the moment the sweeps finish.
#SBATCH --account=devs
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=atgp-ladder
#SBATCH --output=logs/ladder-%x-%j.out

set -uo pipefail
SLOT="${1:?usage: sbatch ladder_job.sbatch <model_slot>}"
cd "$HOME/temporal-agent-probes"
mkdir -p logs runs
source $PROJECT/venvs/atgp2/bin/activate
export HF_HOME=$PROJECT/hf
export VLLM_LOGGING_LEVEL=WARNING

MODEL_ID=$(python -c "import yaml;print(yaml.safe_load(open('configs/models.yaml'))['models']['$SLOT']['id'])")
# A per-job port keeps two checkpoints from colliding if they land on the same node.
PORT=$((8000 + (SLURM_JOB_ID % 1000)))
MAXLEN=32768

echo "== slot=$SLOT model=$MODEL_ID port=$PORT node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Hermes is Qwen2.5's tool-call format. Without --enable-auto-tool-choice vLLM returns tool calls
# as prose, which would show up as a total analytical failure rather than what it is: unparsed
# output. That distinction is the whole point of C4's protocol accounting.
vllm serve "$MODEL_ID" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --disable-log-requests \
  > "logs/vllm-$SLOT-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!

echo "== waiting for vLLM to become healthy"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "== healthy after ${i}0s"; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! vLLM died during startup; last lines:"; tail -30 "logs/vllm-$SLOT-$SLURM_JOB_ID.log"
    exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "!! vLLM never became healthy"; tail -30 "logs/vllm-$SLOT-$SLURM_JOB_ID.log"; kill $VLLM_PID; exit 1; }

export ATGP_VLLM_URL="http://127.0.0.1:$PORT/v1"
export ATGP_VLLM_KEY=EMPTY

# Record the controls C4 requires, as data rather than prose.
python - <<PY > "runs/ladder_${SLOT}_controls.json"
import json, os, subprocess, torch, vllm
print(json.dumps({
    "slot": "$SLOT", "checkpoint": "$MODEL_ID", "engine": f"vllm {vllm.__version__}",
    "torch": torch.__version__, "quantization": "awq",
    "max_model_len": $MAXLEN, "gpu_memory_utilization": 0.90,
    "tool_call_parser": "hermes", "enable_auto_tool_choice": True,
    "temperature": 0.0, "gpu": torch.cuda.get_device_name(0),
    "driver": subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip(),
    "node": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
}, indent=2))
PY

rc=0
for spec in "ladder_m1_a1 A1" "ladder_m1_a3 A3" "ladder_m2_a1 A1" "ladder_m2_a3 A3"; do
  set -- $spec
  echo "== sweep $1 condition $2"
  python scripts/run_matrix.py "$1" --conditions "$2" --models "$SLOT" \
      --out "runs/ladder_$SLOT" --seeds 5 --conc 8 || rc=$?
done

echo "== shutting down vLLM"
kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
echo "== done rc=$rc"
exit $rc
