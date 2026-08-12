#!/usr/bin/env bash
# Push this checkout to the compute host. Source of truth is local; execution is remote (D1).
set -euo pipefail

HOST="${ATGP_HOST:-${ATGP_HOST:?set to your compute host}}"
USER="${ATGP_USER:-$USER}"
REMOTE="${ATGP_REMOTE:-~/temporal-agent-probes}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --delete plus a too-narrow exclude list once wiped data/tasks and data/invariants on the
# remote mid-sweep, because they exist only there. Everything generated lives under data/ or
# runs/, so exclude both wholesale rather than enumerating subdirectories.
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'data/' --exclude 'runs/' --exclude 'results/' \
  "$LOCAL/" "$USER@$HOST:$REMOTE/"

echo "synced -> $USER@$HOST:$REMOTE"
