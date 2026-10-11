#!/bin/bash
# daily-sourcing.sh: the morning sourcing chain for the previous session (com.mtap.sourcing.daily).
#
# Runs after the morning wake because pmset allows only one repeating wake; the
# previous session's Massive bars are published after midnight ET. Sourcing only:
# the ML side (tft-vwap-predict) has its own schedule and reads the exported panel.
#
#   massive_grouped_daily fetch -> ingest-eod -> fetch-splits -> fetch-edgar
#     -> derive-features -> export-tft --include-inference
#
# A failed step stops the chain (later steps depend on it) and is logged.

set -u
source "$(dirname "$0")/lib.sh"
LOG="$MTAP_LOGS/daily-sourcing-$(date +%F).log"

prev_weekday() {  # the weekday before today, New York time
    local dow; dow=$(TZ=America/New_York date +%u)
    local back=1; [ "$dow" = "1" ] && back=3; [ "$dow" = "7" ] && back=2
    TZ=America/New_York date -v-"${back}"d +%F
}

step() {
    echo "$(date '+%H:%M:%S') >>> sourcing-py $*" >> "$LOG"
    if ! sourcing_py "$@" >> "$LOG" 2>&1; then
        mtap_log "daily: FAILED at '$*' (see $LOG)"
        exit 1
    fi
}

caffeinate -is -w $$ &
D="$(prev_weekday)"
mtap_log "daily sourcing for $D (log $LOG)"
[ -n "${MTAP_NAS_URL:-}" ] && "$MTAP_REPO/scripts/storage/mount-nas.sh" >> "$LOG" 2>&1
# Exports go to DATA_ROOT/out; never write them into an unmounted /Volumes folder.
if ! sourcing_py storage check >> "$LOG" 2>&1; then
    mtap_log "daily: storage not ready (see $LOG); not running"
    exit 1
fi
step run massive_grouped_daily --date "$D"
step run massive_grouped_daily --retry-failed
step etrade ingest-eod --days 7
step etrade fetch-splits
step etrade fetch-edgar
step etrade derive-features
step etrade export-tft --include-inference
mtap_log "daily sourcing for $D done"
