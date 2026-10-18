#!/bin/bash
# daily-sourcing.sh: the morning sourcing chain (com.mtap.sourcing.daily).
#
# Runs after the morning wake because pmset allows only one repeating wake; the
# previous session's Massive bars are published after midnight ET. Sourcing only:
# the ML side (tft-vwap-predict) has its own schedule and reads the exported panel.
#
#   massive_grouped_daily for every missed session -> massive-ext gaps (pre-market /
#     after-hours for the same sessions) -> ingest-eod -> fetch-splits -> fetch-edgar
#     -> derive-features -> export-tft --include-inference -> back up the DuckDB
#
# Missed sessions: `sourcing-py daily-gaps` lists every exchange session of the last
# 14 days (never today) without a daily_summary file, so days the Mac slept through or
# a failed run are fetched next time; holidays are not sessions.
#
#   --fill-only                only the Massive fetches (the evening sleep runs this)
#   MTAP_DAILY_CAPTURE_ONLY=1  (.env) same: a capture-only Mac (the iMac) has no
#                              fundamentals DuckDB, so the later steps are skipped there
#   --ingest-only              skip the Massive fetches: the Mac that owns the DuckDB
#                              reads what the capture Mac fetched into the shared NAS out/
#   MTAP_DAILY_INGEST_ONLY=1   (.env) same, for that Mac's scheduled run
#
# A failed day is logged and retried next run (so is a failed extended-hours fetch: its
# sessions stay missing); a failed later step stops the chain (later steps depend on
# it) and is logged. The DuckDB is copied to DATA_ROOT/backup/etrade/ (never opened
# there: SMB has no file locks).

set -u
source "$(dirname "$0")/lib.sh"
LOG="$MTAP_LOGS/daily-sourcing-$(date +%F).log"
FILL_ONLY=0
INGEST_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --fill-only) FILL_ONLY=1 ;;
        --ingest-only) INGEST_ONLY=1 ;;
        *) echo "usage: $0 [--fill-only | --ingest-only]" >&2; exit 2 ;;
    esac
done
[ "${MTAP_DAILY_CAPTURE_ONLY:-0}" = "1" ] && FILL_ONLY=1
[ "${MTAP_DAILY_INGEST_ONLY:-0}" = "1" ] && INGEST_ONLY=1

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

if [ $INGEST_ONLY = 1 ]; then
    [ $FILL_ONLY = 1 ] && { mtap_log "daily: --ingest-only on a capture-only Mac; nothing to do"; exit 0; }
    mtap_log "daily sourcing (ingest only, Massive skipped) (log $LOG)"
else
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
    # Pre-market / after-hours for the sessions that now have daily bars but no
    # extended_hours file (refreshes the weekly common-stock list itself).
    echo "$(date '+%H:%M:%S') >>> sourcing-py massive-ext gaps" >> "$LOG"
    sourcing_py massive-ext gaps --days 14 --max 10 >> "$LOG" 2>&1 \
        || mtap_log "daily: extended-hours fetch failed (see $LOG); retried next run"
fi
if [ $FILL_ONLY = 1 ]; then
    mtap_log "daily sourcing done (Massive only)"
    exit 0
fi
step etrade ingest-eod --days 10
step etrade fetch-splits
step etrade fetch-edgar
step etrade derive-features
step etrade export-tft --include-inference
DB="$(_mtap_abs "${ETRADE_DB_PATH:-state/etrade/fundamentals.duckdb}")"  # as python etrade_config()
if [ -n "${MTAP_DATA_ROOT:-}" ] && [ -f "$DB" ]; then
    BK="$MTAP_DATA/backup/etrade"
    if mkdir -p "$BK" && cp "$DB" "$BK/.fundamentals.duckdb.tmp" && mv -f "$BK/.fundamentals.duckdb.tmp" "$BK/fundamentals.duckdb"; then
        echo "$(date '+%H:%M:%S') backed up $DB to $BK" >> "$LOG"
    else
        rm -f "$BK/.fundamentals.duckdb.tmp"
        mtap_log "daily: DuckDB backup to $BK failed (see $LOG)"
    fi
fi
mtap_log "daily sourcing done"
