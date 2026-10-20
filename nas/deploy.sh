#!/bin/bash
# On the Mac: get a NAS run of massive1s ready, or show how it is going. See nas/README.md.
#
#   ./nas/deploy.sh 2026-09-01 2026-09-30   inputs for that range + copy the source to the
#                                           share; then press Run on the NAS task
#   ./nas/deploy.sh status                  jobs, day files and the newest log lines
#
# Nothing here fetches: the NAS does. Needs the share mounted (default
# /Volumes/Backups/MTap-Data, or MTAP_DATA_ROOT in .env).
set -euo pipefail
# shellcheck source=../scripts/schedule/lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/../scripts/schedule/lib.sh"
die() { echo "deploy: $*" >&2; exit 2; }

if [ -z "${MTAP_DATA_ROOT:-}" ]; then
    export MTAP_DATA_ROOT=/Volumes/Backups/MTap-Data
    MTAP_DATA="$MTAP_DATA_ROOT"
fi
[ -f "$MTAP_DATA/.mtap-data-root" ] || die "$MTAP_DATA is not mounted (no .mtap-data-root)"
M1S="$MTAP_DATA/massive_1s"

status() {
    local p any=0
    for p in "$M1S"/jobs/*/progress.json; do
        [ -f "$p" ] || continue
        any=1
        jq -r '"\(.job)  \(.phase)  \(.done)/\(.tickers) tickers, \(.failed) failed, \(.calls) calls, "
            + "\(.bars / 1e6 | floor)M bars, \(.wire_bytes / 1e9 * 100 | floor / 100) GB, \(.elapsed_s) s"' "$p"
    done
    [ "$any" = 1 ] || echo "no jobs yet"
    echo "day files (source=massive_rest): $(find "$MTAP_DATA/agg_1s" -maxdepth 3 -path '*source=massive_rest/day-*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
    local log
    log="$(ls -t "$M1S"/logs/run-*.log 2>/dev/null | head -1 || true)"
    [ -f "$M1S/logs/task.log" ] && { echo "--- $M1S/logs/task.log (the NAS task)"; tail -n 8 "$M1S/logs/task.log"; }
    [ -n "$log" ] && { echo "--- $log"; tail -n 12 "$log"; }
    return 0
}

if [ "${1:-}" = status ]; then
    status
    exit 0
fi
[ $# -eq 2 ] || die "usage: $0 FROM TO (YYYY-MM-DD) | status"
FROM="$1" TO="$2"
for d in "$FROM" "$TO"; do [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "bad date $d"; done

# The NAS user: MTAP_NAS_USER, else the user the share is mounted as.
NAS_USER="${MTAP_NAS_USER:-$(mount | sed -n 's#^//\([^@]*\)@.* on /Volumes/Backups .*#\1#p' | head -1)}"
[ -n "$NAS_USER" ] || die "set MTAP_NAS_USER in .env (the DSM user that owns the key file)"

sourcing_py massive-1s prepare --fromdate "$FROM" --todate "$TO"
mkdir -p "$MTAP_DATA/nas/massive1s"
rsync -rlt --delete --exclude target/ "$MTAP_REPO/nas/massive1s/" "$MTAP_DATA/nas/massive1s/"
printf 'FROM=%s\nTO=%s\nNAS_USER=%s\nWORKERS=%s\n' "$FROM" "$TO" "$NAS_USER" "${MASSIVE1S_WORKERS:-32}" > "$M1S/.run.args.tmp"
mv "$M1S/.run.args.tmp" "$M1S/run.args"

echo
echo "ready: $FROM .. $TO for NAS user $NAS_USER (check the stock list above: $M1S/inputs/universe.csv)"
echo "next:  DSM > Control Panel > Task Scheduler > massive1s > Run"
echo "watch: ./nas/deploy.sh status"
