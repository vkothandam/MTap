#!/bin/bash
# The E*TRADE collector in the collector container (supervisord program mtap-etrade).
#
# The restart loop of scripts/schedule/mtap-schedule.sh supervise: run `serve etrade`,
# and if it exits restart it after 15 s, doubling to 10 min (back to 15 s after a run of
# over 10 min), so a failing login is not retried in a tight loop. The browser is headed on
# the container's X display, which noVNC (:7900) shows.
set -u
LOGS="${MTAP_STATE_ROOT:-/state}/logs"
mkdir -p "$LOGS"
child=""
trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null; [ -n "$child" ] && wait "$child"; exit 0' TERM INT

for _ in $(seq 1 60); do xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break; sleep 1; done
xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 || echo "$(date '+%F %T') collector: WARNING: no X display $DISPLAY after 60 s"

# Chrome's profile lock names the host (the container's id) and its pid. Chrome clears a
# stale lock of its own host, but a new container (after `build` or a config change) finds
# the old container's lock and refuses the profile as "in use on another computer". A lock
# from another host is always stale: the profile is used by this container only.
profile="${CHROME_PROFILE_DIR:-${MTAP_STATE_ROOT:-/state}/etrade/chrome-profile}"
lock=$(readlink "$profile/SingletonLock" 2>/dev/null)
if [ -n "$lock" ] && [ "${lock%-*}" != "$(hostname)" ]; then
    echo "$(date '+%F %T') collector: removing a stale Chrome profile lock from host ${lock%-*}"
    rm -f "$profile/SingletonLock" "$profile/SingletonCookie" "$profile/SingletonSocket"
fi

cd /app/node || exit 1
backoff=15
while :; do
    started=$(date +%s)
    log="$LOGS/collector-$(date +%F).log"
    echo "$(date '+%F %T') collector: starting (log $log)"
    node --import tsx src/cli.ts serve etrade >> "$log" 2>&1 &
    child=$!
    wait "$child"; code=$?
    child=""
    if [ $(( $(date +%s) - started )) -gt 600 ]; then backoff=15; fi
    echo "$(date '+%F %T') collector: exited with $code; restarting in ${backoff}s"
    sleep "$backoff" & wait $!
    backoff=$(( backoff * 2 > 600 ? 600 : backoff * 2 ))
done
