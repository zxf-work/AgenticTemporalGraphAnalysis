#!/usr/bin/env bash
# Sync THEN run, in one step, and prove the remote is running the committed code.
#
# Rationale: the M3 margin fix was committed locally and never synced, so a re-run silently used
# the superseded config, produced byte-identical worlds, and cost $11.59 for nothing. Nothing in
# the pipeline compared "code that ran" against "code I committed". Resolving to remember is not
# a fix; removing the opportunity to forget is.
set -euo pipefail

HOST="${ATGP_HOST:-${ATGP_HOST:?set to your compute host}}"
USER="${ATGP_USER:-$USER}"
REMOTE="${ATGP_REMOTE:-~/temporal-agent-probes}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "$(git -C "$LOCAL" status --porcelain 2>/dev/null)" ]]; then
  echo "REFUSING: working tree is dirty. Commit first so the run is traceable to a commit." >&2
  git -C "$LOCAL" status --short >&2
  exit 1
fi
COMMIT="$(git -C "$LOCAL" rev-parse --short HEAD)"

bash "$LOCAL/scripts/sync.sh" >/dev/null
ssh -o BatchMode=yes "$USER@$HOST" "printf '%s' '$COMMIT' > $REMOTE/.synced_commit"

REMOTE_COMMIT="$(ssh -o BatchMode=yes "$USER@$HOST" "cat $REMOTE/.synced_commit")"
if [[ "$REMOTE_COMMIT" != "$COMMIT" ]]; then
  echo "REFUSING: remote stamp $REMOTE_COMMIT != local HEAD $COMMIT" >&2
  exit 1
fi
echo "synced and verified at commit $COMMIT"

# Activation differs per host: compute-host runs a conda env and needs API keys; gpu-cluster runs a uv venv
# and needs none. Keeping it in one variable means both hosts go through the same guard.
ACTIVATE="${ATGP_ACTIVATE:-source ~/miniconda3/etc/profile.d/conda.sh && conda activate atgp && source ~/.atgp_env}"
ssh -o BatchMode=yes "$USER@$HOST" "bash -lc '$ACTIVATE && cd $REMOTE && $*'"
