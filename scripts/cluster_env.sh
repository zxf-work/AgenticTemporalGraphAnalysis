# Point the sync/run guard at gpu-cluster. Usage:  source scripts/gpu-cluster_env.sh
# gpu-cluster serves open-weight subjects locally (milestone C4); it holds no API keys by design.
export ATGP_HOST=gpu-cluster
export ATGP_USER=$USER
export ATGP_REMOTE='~/temporal-agent-probes'
export ATGP_ACTIVATE='source $PROJECT/venvs/atgp2/bin/activate && export HF_HOME=$PROJECT/hf'
