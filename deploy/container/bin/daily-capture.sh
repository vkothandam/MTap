#!/bin/bash
# daily-capture.sh: the Massive daily fetches, in the py container (jobs.sh daily / evening).
#
# The capture branch of scripts/schedule/daily-sourcing.sh (what the iMac runs with
# MTAP_DAILY_CAPTURE_ONLY=1): grouped daily bars for every missed session of the last 14
# days, retry the failed ones, then pre-market / after-hours for the same sessions. The
# fundamentals steps (ingest-eod ... export-tft) need the DuckDB, which is not on this host.
#
# A failed day is logged and fetched next run; a failed --retry-failed stops the run.
set -u
ts() { date '+%H:%M:%S'; }

echo "$(ts) >>> sourcing-py storage check"
sourcing-py storage check || { echo "$(ts) daily: storage not ready; not running"; exit 1; }

if ! GAPS="$(sourcing-py daily-gaps --days 14 --max 10)"; then
    echo "$(ts) daily: FAILED listing missed sessions"
    exit 1
fi
GAPS="$(echo $GAPS)"  # one line
echo "$(ts) daily: sessions to fetch: ${GAPS:-none}"
for D in $GAPS; do
    echo "$(ts) >>> sourcing-py run massive_grouped_daily --date $D"
    sourcing-py run massive_grouped_daily --date "$D" \
        || echo "$(ts) daily: Massive fetch for $D failed; retried next run"
done
echo "$(ts) >>> sourcing-py run massive_grouped_daily --retry-failed"
sourcing-py run massive_grouped_daily --retry-failed || { echo "$(ts) daily: FAILED at --retry-failed"; exit 1; }
echo "$(ts) >>> sourcing-py massive-ext gaps --days 14 --max 10"
sourcing-py massive-ext gaps --days 14 --max 10 \
    || echo "$(ts) daily: extended-hours fetch failed; retried next run"
echo "$(ts) daily: done"
