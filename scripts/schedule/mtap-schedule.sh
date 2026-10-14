#!/bin/bash
# mtap-schedule.sh: start/stop the E*TRADE watchlist collector, and the evening sleep.
# Called by the com.mtap.* LaunchAgents (scripts/schedule/setup-schedule.sh), or by hand.
#
#   start      mount the NAS, check storage and Redis, then run the collector under a
#              supervisor (restarts it if it dies) with caffeinate holding the Mac awake
#   stop       stop the collector
#   sleep      stop the collector, flush the storage consumer to the NAS, log the day's
#              summary, fill missed daily bars, wait for the Massive collector to finish
#              its window and upload, combine its files into one per day (if installed),
#              then put the Mac to sleep
#   status     collector, storage consumer, Massive collector, Redis, LaunchAgents, power
#   validate   check node/npm/uv/redis, .env and the browser before the first scheduled run
#   run-store  (com.mtap.store) the storage consumer, in the foreground
#   run-massive (com.mtap.massive) the Massive per-second collector, in the foreground
#   supervise  (internal) the restart loop that `start` launches
#
# Ported from MBin's trading-schedule.sh. Credentials live only in the repo .env.

set -u
source "$(dirname "$0")/lib.sh"

PID_FILE="$MTAP_STATE/collector.pid"
STOP_FLAG="$MTAP_STATE/collector.stop"
COLLECTOR_LOG() { echo "$MTAP_LOGS/collector-$(date '+%Y-%m-%d').log"; }
UID_N="$(id -u)"

log_env() {
    mtap_log "repo=$MTAP_REPO state=$MTAP_STATE data=$MTAP_DATA"
    mtap_log "node=$(command -v node || echo MISSING) $(node --version 2>/dev/null) uv=$(command -v uv || echo MISSING)"
    mtap_log "BUS_TRANSPORT=${BUS_TRANSPORT:-unset} REDIS_URL=${REDIS_URL:-default} INTRADAY_FEEDS=${INTRADAY_FEEDS:-default} HEADLESS=$HEADLESS"
}

validate() {
    local errors=0
    for tool in node npm uv redis-cli; do
        command -v "$tool" >/dev/null 2>&1 || { mtap_log "ERROR: $tool not found in PATH ($PATH)"; errors=$((errors + 1)); }
    done
    [ -f "$MTAP_REPO/.env" ] || { mtap_log "ERROR: $MTAP_REPO/.env missing (copy .env.example)"; errors=$((errors + 1)); }
    [ -x "$MTAP_REPO/node/node_modules/.bin/tsx" ] || { mtap_log "ERROR: node deps missing (cd node && npm ci)"; errors=$((errors + 1)); }
    [ -d "$MTAP_REPO/python/.venv" ] || { mtap_log "ERROR: python env missing (cd python && uv sync)"; errors=$((errors + 1)); }
    [ "${BUS_TRANSPORT:-}" = "redis" ] || { mtap_log "ERROR: BUS_TRANSPORT must be redis in .env (the storage consumer reads Redis)"; errors=$((errors + 1)); }
    [ -n "${ET_USERNAME:-}" ] && [ -n "${ET_PASSWORD:-}" ] || { mtap_log "ERROR: ET_USERNAME / ET_PASSWORD not set in .env"; errors=$((errors + 1)); }
    [ "${FEATURE_SCRAPING:-true}" != "false" ] || mtap_log "WARNING: FEATURE_SCRAPING=false: the collector will not poll"
    if [ -z "$(ls -d "$PUPPETEER_CACHE_DIR"/chrome* 2>/dev/null)" ] && [ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
        mtap_log "ERROR: no puppeteer browser in $PUPPETEER_CACHE_DIR (cd node && npx puppeteer browsers install chrome)"
        errors=$((errors + 1))
    fi
    local py_roots
    py_roots="$(cd "$MTAP_REPO/python" && uv run --quiet python -c 'from sourcing_py.common import config; print(config.state_root()); print(config.data_root())' 2>/dev/null)"
    if [ -n "$py_roots" ] && [ "$py_roots" != "$(printf '%s\n%s' "$MTAP_STATE" "$MTAP_DATA")" ]; then
        mtap_log "ERROR: Python resolves the roots differently ($(echo $py_roots)); set MTAP_STATE_ROOT / MTAP_DATA_ROOT in .env, not only in settings.toml"
        errors=$((errors + 1))
    fi
    if mtap_redis_cli ping >/dev/null 2>&1; then :; else mtap_log "WARNING: Redis not answering at ${REDIS_URL:-redis://127.0.0.1:6379}"; fi
    if pgrep -f "MBin/.*server/server.js" >/dev/null 2>&1; then
        mtap_log "WARNING: an MBin server is running; it must not scrape the same E*TRADE session"
    fi
    [ $errors -eq 0 ]
}

collector_pid() {  # the live supervisor's pid; a pid file left by a crash or power loss is ignored
    [ -f "$PID_FILE" ] || return 1
    local pid; pid="$(cat "$PID_FILE")"
    ps -p "$pid" -o command= 2>/dev/null | grep -q "mtap-schedule.sh supervise" && echo "$pid"
}

# The restart loop. Runs the collector in the foreground, restarts it with backoff if it
# exits, and holds a caffeinate assertion for as long as it lives.
supervise() {
    local child="" backoff=15 started
    trap 'touch "$STOP_FLAG"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM INT
    caffeinate -is -w $$ &
    cd "$MTAP_REPO/node" || exit 1
    while [ ! -f "$STOP_FLAG" ]; do
        started=$(date +%s)
        mtap_log "collector: starting (log $(COLLECTOR_LOG))"
        node --import tsx src/cli.ts serve etrade >> "$(COLLECTOR_LOG)" 2>&1 &
        child=$!
        wait "$child"; local code=$?
        # `wait` returns early when the trap fires; wait again for the real exit.
        while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done
        child=""
        [ -f "$STOP_FLAG" ] && break
        if [ $(( $(date +%s) - started )) -gt 600 ]; then backoff=15; fi
        mtap_log "collector: exited with $code; restarting in ${backoff}s"
        sleep "$backoff" & wait $!
        backoff=$(( backoff * 2 > 600 ? 600 : backoff * 2 ))
    done
    mtap_log "collector: supervisor exiting"
}

start() {
    caffeinate -i -w $$ &   # stay awake through validate/mount until the supervisor holds its own
    mtap_log "========== START =========="
    log_env
    if pid="$(collector_pid)"; then mtap_log "collector already running (supervisor $pid)"; return 0; fi
    validate || { mtap_log "validation failed; not starting (see above)"; return 1; }

    if [ -n "${MTAP_NAS_URL:-}" ]; then
        "$MTAP_REPO/scripts/storage/mount-nas.sh" >> "$MTAP_SCHEDULE_LOG" 2>&1 || mtap_log "WARNING: NAS mount failed; segments stay local until it is back"
    fi
    sourcing_py storage check >> "$MTAP_SCHEDULE_LOG" 2>&1 || mtap_log "WARNING: storage check failed (see above); collection continues, uploads wait"

    # Redis, the storage consumer and the Massive collector are KeepAlive LaunchAgents;
    # make sure they are up.
    for job in com.mtap.redis com.mtap.store com.mtap.massive; do
        launchctl print "gui/$UID_N/$job" >/dev/null 2>&1 && launchctl kickstart "gui/$UID_N/$job" >/dev/null 2>&1
    done
    # One that is alive but silent (stuck on a stale NAS mount after sleep) is restarted.
    # Both write their heartbeat every 10 s, idle or not.
    local job hb
    for job in "com.mtap.store:$MTAP_STATE/intraday/store.heartbeat" "com.mtap.massive:$MTAP_STATE/massive/collector.heartbeat"; do
        hb="${job#*:}"; job="${job%%:*}"
        if [ -f "$hb" ] && [ $(( $(date +%s) - $(stat -f %m "$hb") )) -gt 120 ] \
            && launchctl print "gui/$UID_N/$job" >/dev/null 2>&1; then
            mtap_log "$job heartbeat is stale; restarting it"
            launchctl kickstart -k "gui/$UID_N/$job" >/dev/null 2>&1
        fi
    done
    for _ in 1 2 3 4 5 6 7 8 9 10; do mtap_redis_cli ping >/dev/null 2>&1 && break; sleep 1; done
    mtap_redis_cli ping >/dev/null 2>&1 || mtap_log "WARNING: Redis is not up; the collector will retry publishing"

    rm -f "$STOP_FLAG"
    # mtap_log already writes to schedule.log; only stray stderr goes there directly.
    nohup "$0" supervise >/dev/null 2>> "$MTAP_SCHEDULE_LOG" &
    echo $! > "$PID_FILE"
    sleep 5
    if pid="$(collector_pid)"; then
        mtap_log "collector started (supervisor $pid); OTP page on ${OTP_HOST:-127.0.0.1}:${OTP_PORT:-8787} if a login needs a code"
    else
        mtap_log "ERROR: supervisor exited immediately; see $(COLLECTOR_LOG)"
        return 1
    fi
}

stop() {
    mtap_log "========== STOP =========="
    touch "$STOP_FLAG"
    if pid="$(collector_pid)"; then
        kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 1 30); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 1; done
        if ps -p "$pid" >/dev/null 2>&1; then
            mtap_log "supervisor $pid still running after 30 s; killing"
            pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
        fi
    else
        mtap_log "collector not running (no live supervisor)"
    fi
    rm -f "$PID_FILE"
    # Belt and braces: a collector left behind by a killed supervisor.
    local stray; stray="$(pgrep -f "$MTAP_REPO/node.*src/cli.ts serve etrade|tsx src/cli.ts serve etrade" 2>/dev/null)"
    if [ -n "$stray" ]; then
        mtap_log "stopping leftover collector process(es): $stray"
        kill -TERM $stray 2>/dev/null; sleep 5; kill -KILL $stray 2>/dev/null
    fi
    mtap_log "collector stopped"
}

# Close the store's open segments now (it would otherwise wait idle_close_s), let it
# upload them, and wait until nothing is left locally or 90 s pass.
flush_store() {
    if launchctl print "gui/$UID_N/com.mtap.store" >/dev/null 2>&1; then
        mtap_log "flushing the storage consumer (it closes, uploads and is restarted by launchd)"
        launchctl kill TERM "gui/$UID_N/com.mtap.store" 2>/dev/null
    else
        sourcing_py intraday upload --drain >> "$MTAP_SCHEDULE_LOG" 2>&1
    fi
    for _ in $(seq 1 18); do
        sleep 5
        [ -z "$(find "$MTAP_STATE/segments" -type f \( -name '*.arrows' -o -name '*.parquet' \) 2>/dev/null | head -1)" ] && break
    done
    local left; left="$(find "$MTAP_STATE/segments" -type f 2>/dev/null | wc -l | tr -d ' ')"
    [ "$left" = "0" ] && mtap_log "all segments uploaded" || mtap_log "WARNING: $left segment file(s) still local (NAS down?); they upload when it is back"
}

go_to_sleep() {
    caffeinate -i -w $$ &   # stay awake while the store flushes to the NAS
    mtap_log "========== SLEEP =========="
    stop
    flush_store
    sourcing_py intraday summary --date "$(TZ=America/New_York date +%F)" 2>&1 | tee -a "$MTAP_LOGS/daily-summary.log" >> "$MTAP_SCHEDULE_LOG"
    # Daily bars a morning run missed (never today's: it is final only after 20:00 ET).
    [ -n "${MASSIVE_API_KEY:-}" ] && "$MTAP_REPO/scripts/schedule/daily-sourcing.sh" --fill-only
    wait_massive
    find "$MTAP_LOGS" -name '*.log' -mtime +14 -delete 2>/dev/null
    mtap_log "putting the Mac to sleep"
    pmset sleepnow
}

# The Massive feed is 15 min behind, so its window closes after the E*TRADE collector
# stops (16:21 ET, so the closing prints arrive). Wait for it to close its last segment
# and upload, up to 20 min (the pmset backstop sleeps the Mac at 13:40 PT regardless).
wait_massive() {
    launchctl print "gui/$UID_N/com.mtap.massive" >/dev/null 2>&1 || return 0
    mtap_log "waiting for the Massive collector to finish its window and upload"
    if sourcing_py massive-ws wait-idle --timeout 1200 >> "$MTAP_SCHEDULE_LOG" 2>&1; then
        mtap_log "Massive collector idle, all uploaded"
    else
        mtap_log "WARNING: Massive collector not idle/uploaded in time; its segments upload on the next run"
    fi
    sourcing_py massive-ws status 2>&1 | tee -a "$MTAP_LOGS/daily-summary.log" >> "$MTAP_SCHEDULE_LOG"
    # One file per day on the NAS (safe to interrupt; the next evening finishes it).
    sourcing_py massive-ws compact 2>&1 | tee -a "$MTAP_LOGS/daily-summary.log" >> "$MTAP_SCHEDULE_LOG"
}

status() {
    echo "=== MTap collector ==="
    if pid="$(collector_pid)"; then echo "  running (supervisor $pid, since $(ps -p "$pid" -o lstart= 2>/dev/null))"; else echo "  not running"; fi
    echo ""
    echo "=== LaunchAgents ==="
    launchctl list | grep -E "com\.mtap\." || echo "  (none loaded)"
    if launchctl list | grep -qE "com\.trading\."; then echo "  WARNING: MBin com.trading.* agents are loaded too"; fi
    echo ""
    echo "=== Redis ==="
    if mtap_redis_cli ping >/dev/null 2>&1; then
        echo "  up; INTRADAY_FEED length $(mtap_redis_cli XLEN mtap:evt:INTRADAY_FEED), POLLED_DATA $(mtap_redis_cli XLEN mtap:evt:POLLED_DATA)"
        echo "  memory $(mtap_redis_cli INFO memory | grep -E '^used_memory_human' | cut -d: -f2 | tr -d '\r')"
    else
        echo "  not answering at ${REDIS_URL:-redis://127.0.0.1:6379}"
    fi
    echo ""
    echo "=== Storage consumer and heartbeats ==="
    sourcing_py intraday status 2>&1 | sed 's/^/  /'
    echo ""
    echo "=== Massive per-second collector ==="
    sourcing_py massive-ws status 2>&1 | sed 's/^/  /'
    echo ""
    echo "=== Power schedule ==="
    pmset -g sched | sed 's/^/  /'
    echo ""
    echo "=== Recent schedule log ==="
    tail -10 "$MTAP_SCHEDULE_LOG" 2>/dev/null | sed 's/^/  /'
}

run_store() {  # exec the venv's entry point directly so launchd's SIGTERM reaches Python
    cd "$MTAP_REPO/python" || exit 1
    [ -x .venv/bin/sourcing-py ] && exec .venv/bin/sourcing-py intraday store --follow
    exec uv run --quiet sourcing-py intraday store --follow
}

run_massive() {  # same, for com.mtap.massive; it connects only inside its window
    cd "$MTAP_REPO/python" || exit 1
    [ -x .venv/bin/sourcing-py ] && exec .venv/bin/sourcing-py massive-ws follow
    exec uv run --quiet sourcing-py massive-ws follow
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    sleep) go_to_sleep ;;
    status) status ;;
    validate) log_env; if validate; then echo "environment OK"; else echo "validation failed; see $MTAP_SCHEDULE_LOG"; exit 1; fi ;;
    run-store) run_store ;;
    run-massive) run_massive ;;
    supervise) supervise ;;
    *)
        echo "usage: $0 {start|stop|sleep|status|validate|run-store|run-massive}"
        exit 2
        ;;
esac
