#!/bin/bash
# backup-runs.sh: copy the trained model runs (ml/runs, not in git: checkpoints and test
# predictions, GBs) to the NAS, so a finished training is never only on this Mac.
#
#   ./scripts/storage/backup-runs.sh                 every run, new or changed files only
#   ./scripts/storage/backup-runs.sh --after PID     wait for training process PID to exit
#                                                    first (queue it behind a run)
#
# Destination: $MTAP_RUNS_BACKUP, default /Volumes/Backups/PProjects/MTap/ml/runs
# (smb://192.168.1.100/Backups/PProjects/MTap). Nothing is ever deleted there. A STOP file
# is not copied, so a paused run's copy never looks paused-on-purpose.

set -eu
REPO="$(cd "$(dirname "$0")/../.." && pwd -P)"
SRC="$REPO/ml/runs/"
DEST="${MTAP_RUNS_BACKUP:-/Volumes/Backups/PProjects/MTap/ml/runs}"

case "${1:-}" in
    "") ;;
    --after)
        pid="${2:?usage: $0 [--after PID]}"
        echo "$(date '+%F %T') waiting for pid $pid"
        while kill -0 "$pid" 2>/dev/null; do sleep 60; done ;;
    *) echo "usage: $0 [--after PID]" >&2; exit 2 ;;
esac

[ -d "$(dirname "$DEST")" ] || { echo "backup-runs: $(dirname "$DEST") is not mounted" >&2; exit 1; }
mkdir -p "$DEST"
echo "$(date '+%F %T') $SRC -> $DEST"
# SMB: no owners, groups or permissions; times are kept so a rerun copies only what changed
rsync -rt --no-perms --no-owner --no-group --exclude STOP --stats "$SRC" "$DEST/" | tail -4
echo "$(date '+%F %T') done"
