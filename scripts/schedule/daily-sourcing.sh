#!/bin/bash
# daily-sourcing.sh: the morning sourcing chain (com.mtap.sourcing.daily).
#
# Runs after the morning wake because pmset allows only one repeating wake; the
# previous session's Massive bars are published after midnight ET. Sourcing only:
# the ML side (tft-vwap-predict) has its own schedule and reads the exported panel.
#
#   massive_grouped_daily for every missed session -> ingest-eod -> fetch-splits
#     -> fetch-edgar -> derive-features -> export-tft --include-inference
#
# Missed sessions: `sourcing-py daily-gaps` lists every exchange session of the last
# 14 days (never today) without a daily_summary file, so days the Mac slept through or
# a failed run are fetched next time; holidays are not sessions.
#
#   --fill-only                only the Massive fetches (the evening sleep runs this)
#   MTAP_DAILY_CAPTURE_ONLY=1  (.env) same: a capture-only Mac (the iMac) has no
#                              fundamentals DuckDB, so the later steps are skipped there
#
# A failed day is logged and retried next run; a failed later step stops the chain
# (later steps depend on it) and is logged.

set -u
source "$(dirname "$0")/lib.sh"
LOG="$MTAP_LOGS/daily-sourcing-$(date +%F).log"
FILL_ONLY=0
[ "${1:-}" = "--fill-only" ] && FILL_ONLY=1
[ "${MTAP_DAILY_CAPTURE_ONLY:-0}" = "1" ] && FILL_ONLY=1

step() {
    echo "$(date '+%H:%M:%S') >>> sourcing-py $*" >> "$LOG"
    if ! sourcing_py "$@" >> "$LOG" 2>&1; then
        mtap_log "daily: FAILED at '$*' (see $LOG)"
        exit 1
    fi
}

caffeinate -is -w $$ &
[ -n "${MTAP_NAS_URL:-}" ] && "$MTAP_REPO/scripts/storage/mount-nas.sh" >> "$LOG" 2>&1
# Exports go to DATA_ROOT/out; never write them into an unmounted /Volumes folder.
if ! sourcing_py storage check >> "$LOG" 2>&1; then
    mtap_log "daily: storage not ready (see $LOG); not running"
    exit 1
fi

if ! GAPS="$(sourcing_py daily-gaps --days 14 --max 10 2>> "$LOG")"; then
    mtap_log "daily: FAILED listing missed sessions (see $LOG)"
    exit 1
fi
GAPS="$(echo $GAPS)"  # one line
mtap_log "daily sourcing$([ $FILL_ONLY = 1 ] && echo ' (Massive only)'): sessions to fetch: ${GAPS:-none} (log $LOG)"
for D in $GAPS; do
    echo "$(date '+%H:%M:%S') >>> sourcing-py run massive_grouped_daily --date $D" >> "$LOG"
    sourcing_py run massive_grouped_daily --date "$D" >> "$LOG" 2>&1 \
        || mtap_log "daily: Massive fetch for $D failed (see $LOG); retried next run"
done
step run massive_grouped_daily --retry-failed
if [ $FILL_ONLY = 1 ]; then
    mtap_log "daily sourcing done (Massive only)"
    exit 0
fi
step etrade ingest-eod --days 10
step etrade fetch-splits
step etrade fetch-edgar
step etrade derive-features
step etrade export-tft --include-inference
mtap_log "daily sourcing done"
