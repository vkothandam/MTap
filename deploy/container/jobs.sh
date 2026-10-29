#!/bin/bash
# jobs.sh: MTap's sourcing jobs on any Docker host (the NAS now, a cloud VM later).
# Needs only bash, GNU find and docker; everything else runs in the containers (compose.yml).
#
#   MTAP_ENV_FILE=/path/host.env  jobs.sh JOB
#
#   build     build both images and pull redis (first time, and after a code push)
#   start     the morning start: the roles in MTAP_ROLES
#               etrade      Redis + the storage consumer + the E*TRADE collector
#               massive-ws  the Massive per-second collector (exits by itself after 16:21 ET)
#   daily     the Massive daily fetches for every missed session (bin/daily-capture.sh)
#   evening   stop the collector, flush the store, the day's summary, the daily fill, wait
#             for massive-ws to end its window, its status and compact, prune old logs
#   stop      stop every container
#   status    containers, heartbeats, newest files, recent errors -> STATE/logs/status.txt
#   ws-test   a 2-minute massive-ws connection, outside its window (a trial; refused
#             while the massive-ws service runs: one connection per key)
#   selftest  what a py container sees: its user, Python's path, the package and its CLI
#   morning   forecast-night if no host has finished it, 09:29:30 ET precheck, 09:30 prices
#             (watchlist, else ChartIQ price-only; token file only, never logs in, stops on the
#             first 401/403/429), the open until 09:32, export-next + predict px + screener-score,
#             09:45 Massive full pre-market, export-next + predict full + screener-score. Every
#             step is time-boxed. Takes NO jobs lock; a pid file in STATE keeps a second out
#   night     forecast-night for the next session (claim file in STATE/forecast: once per day
#             across hosts; refused before 20:20 ET)
#   inputs-backfill  the 5-year inputs, resumable and idempotent: back up out/extended_hours
#             once, massive-ext refetch from MTAP_BACKFILL_FROM with minute bars, tickers_cs
#             refresh, quarter-end shares, the economy series. Queue it after the evening job.
#             Finished quarters are skipped; MTAP_BACKFILL_PARALLEL=N fetches N quarters at once;
#             MTAP_BACKFILL_TO (default yesterday) leaves the later quarters to another host
#   run ARGS  sourcing-py ARGS in a one-off py container
#   queue     run the job named in STATE/run-next (one of build start daily evening stop
#             status ws-test selftest morning night inputs-backfill; no arguments), then delete the file; nothing if it
#             is absent. Every queue run also runs `watch` first.
#   watch     in market hours (Mon-Fri ET), a role's heartbeats older than
#             MTAP_WATCH_STALE_MIN (10) minutes are a problem; exit 1, with the reason in
#             STATE/logs/alert.txt, when a new problem appears or all have cleared (once per
#             incident: the host mails a non-zero exit)
#
# The env file sets the host paths (MTAP_DATA_DIR, MTAP_STATE_DIR, MTAP_SECRETS_FILE), the
# data owner (MTAP_UID, MTAP_GID), MTAP_ROLES and the containers' settings. MTAP_ENV_OVERRIDE,
# if it names a file, is read after it (host-level values only: compose passes MTAP_ENV_FILE
# to the containers). Output goes to stdout; the host wrapper (deploy/synology/task.sh)
# appends it to STATE/logs/jobs.log. Status lines also go to STATE/logs/daily-summary.log.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
JOB="${1:-}"
die() { echo "jobs: $*" >&2; exit 2; }

[ -n "${MTAP_ENV_FILE:-}" ] && [ -f "$MTAP_ENV_FILE" ] || die "MTAP_ENV_FILE must name the host's env file"
set -a
# shellcheck disable=SC1090
. "$MTAP_ENV_FILE"
# shellcheck disable=SC1090
if [ -n "${MTAP_ENV_OVERRIDE:-}" ] && [ -f "$MTAP_ENV_OVERRIDE" ]; then . "$MTAP_ENV_OVERRIDE"; fi
set +a
for v in MTAP_DATA_DIR MTAP_STATE_DIR MTAP_SECRETS_FILE MTAP_UID MTAP_GID; do
    [ -n "${!v:-}" ] || die "$v is not set in $MTAP_ENV_FILE"
done
export MTAP_ENV_FILE
LOGS="$MTAP_STATE_DIR/logs"
SUMMARY_LOG="$LOGS/daily-summary.log"
ALL=(--profile etrade --profile massive-ws --profile job)

if docker compose version >/dev/null 2>&1; then DC=(docker compose); else DC=(docker-compose); fi
compose() { "${DC[@]}" -f "$HERE/compose.yml" "$@"; }
py() { compose run --rm -T job sourcing-py "$@"; }   # a one-off py container
log() { echo "$(date '+%F %T') [$JOB] $*"; }
has_role() { case " ${MTAP_ROLES:-} " in *" $1 "*) return 0 ;; esac; return 1; }
has_key() { grep -Eq "^(export )?MASSIVE_API_KEY=['\"]?[^'\" ]" "$MTAP_SECRETS_FILE"; }
data_ready() { [ -f "$MTAP_DATA_DIR/.mtap-data-root" ]; }
running() { [ -n "$(compose "${ALL[@]}" ps -q "$1" 2>/dev/null)" ]; }

# The state folders, owned by the data owner (this runs as root on the NAS).
prepare_state() {
    local d
    for d in "" logs etrade intraday segments segments_massive massive rate failures; do
        mkdir -p "$MTAP_STATE_DIR/$d" || die "cannot create $MTAP_STATE_DIR/$d"
        chown "$MTAP_UID:$MTAP_GID" "$MTAP_STATE_DIR/$d" 2>/dev/null || true
    done
}

# Files this script wrote as root go back to the data owner.
fix_owner() {
    find "$LOGS" -maxdepth 1 -type f ! -user "$MTAP_UID" -exec chown "$MTAP_UID:$MTAP_GID" {} + 2>/dev/null || true
}

check_secrets() {
    [ -f "$MTAP_SECRETS_FILE" ] || die "no secrets file $MTAP_SECRETS_FILE (deploy/synology/README.md, one-time setup)"
    [ "$(stat -c %a "$MTAP_SECRETS_FILE")" = 600 ] || die "$MTAP_SECRETS_FILE must be chmod 600"
}

build() {
    log "building the images"
    compose "${ALL[@]}" build || { log "ERROR: build failed"; return 1; }
    compose "${ALL[@]}" pull redis || { log "ERROR: pulling redis failed"; return 1; }
    log "images built: $(docker image ls --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E '^(mtap-|redis:)' | tr '\n' ' ')"
}

start() {
    log "========== START =========="
    data_ready || { log "ERROR: $MTAP_DATA_DIR has no .mtap-data-root; not starting"; return 1; }
    local profiles=() services=() r
    for r in ${MTAP_ROLES:-}; do
        case "$r" in
            etrade) profiles+=(--profile etrade); services+=(redis store etrade) ;;
            massive-ws)
                if has_key; then profiles+=(--profile massive-ws); services+=(massive-ws)
                else log "massive-ws: no MASSIVE_API_KEY in $MTAP_SECRETS_FILE; not starting"; fi ;;
            *) log "WARNING: unknown role '$r' in MTAP_ROLES" ;;
        esac
    done
    [ ${#services[@]} -gt 0 ] || { log "no roles to start (MTAP_ROLES='${MTAP_ROLES:-}')"; return 0; }
    py storage check || log "WARNING: storage check failed (see above); collection continues, uploads wait"
    compose "${profiles[@]}" up -d "${services[@]}" || { log "ERROR: compose up failed"; return 1; }
    log "started ${services[*]} (data $MTAP_DATA_DIR)"
    if has_role etrade; then
        log "E*TRADE login: watch on http://${OTP_HOST:-<host>}:${MTAP_NOVNC_PORT:-7900}; a texted code goes to http://${OTP_HOST:-<host>}:${OTP_PORT:-8787}/auth/otp"
    fi
}

daily() {
    data_ready || { log "ERROR: $MTAP_DATA_DIR has no .mtap-data-root; not running"; return 1; }
    has_key || { log "daily: no MASSIVE_API_KEY in $MTAP_SECRETS_FILE; nothing to do"; return 0; }
    local dlog; dlog="$LOGS/daily-sourcing-$(date +%F).log"
    log "daily: Massive fetches (log $dlog)"
    if compose run --rm -T job /app/deploy/container/bin/daily-capture.sh >> "$dlog" 2>&1; then
        log "daily: $(tail -1 "$dlog")"
    else
        log "daily: FAILED (see $dlog)"
        return 1
    fi
}

# ---- the morning and the night forecast -------------------------------------------------------
# ET clock helpers: seconds since midnight ET, and a bounded wait until an ET time.
et_now() { TZ=America/New_York date +%H:%M:%S; }
et_secs() { local t; t="$(et_now)"; echo $((10#${t:0:2} * 3600 + 10#${t:3:2} * 60 + 10#${t:6:2})); }
wait_et() {  # wait_et HH:MM:SS: sleep until then; returns 1 at once if it is more than 2 h away
    local h m s want now
    IFS=: read -r h m s <<< "$1"
    want=$((10#$h * 3600 + 10#$m * 60 + 10#${s:-0})); now="$(et_secs)"
    [ "$now" -ge "$want" ] && return 0
    [ $((want - now)) -le 7200 ] || return 1
    sleep $((want - now))
}
# A step that may not run longer than $1 seconds: step_box SECS NAME cmd...
step_box() {
    local secs="$1" name="$2"; shift 2
    log "morning: $name"
    if timeout "$secs" "$@" >> "$MLOG" 2>&1; then log "morning: $name ok"; return 0; fi
    log "morning: $name FAILED or timed out after ${secs}s (see $MLOG)"; return 1
}
# ML predict and the screener run in the job container (MTAP_PREDICT_CMD in the env file names
# the command, default "uv run --project ml tft-vwap-predict"); the image must carry ml/ and
# the model, see deploy/synology/README.md.
predict() {  # predict STATE [symbols-file]
    local extra=""; [ -n "${2:-}" ] && extra="--symbols-file $2"
    compose run --rm -T job sh -c "${MTAP_PREDICT_CMD:-uv run --project ml tft-vwap-predict} \"\${MTAP_FORECAST_RUN_DIR:?set MTAP_FORECAST_RUN_DIR}\" --state $1 $extra"
}
# The screener reads that state's forecast: the main run dir is full, `<run>~px` is px (see
# ml/README.md). MTAP_SCREENER_CMD (default "uv run --project ml screener-score") and
# MTAP_SCREENER_MODEL (runs/screener/<id>) come from the env file; a px pass sees the
# shortlist subset, the full pass the whole forecast.
screener() {  # screener STATE DATE [symbols-file]
    local st="$1" d="$2" sf="${3:-}" fdir suffix="" extra=""
    fdir='${MTAP_FORECAST_RUN_DIR}'; [ "$st" = px ] && fdir='${MTAP_FORECAST_RUN_DIR}~px'
    [ -n "$sf" ] && { suffix="_subset"; extra="--symbols-file $sf"; }
    compose run --rm -T job sh -c "${MTAP_SCREENER_CMD:-uv run --project ml screener-score} --date $d --state $st \
        --model \"\${MTAP_SCREENER_MODEL:?set MTAP_SCREENER_MODEL}\" \
        --prices /data/out/premarket_live/prices-$d.jsonl \
        $([ "$st" = full ] && echo "--pm /data/out/premarket_live/premarket-$d.jsonl") \
        --forecast \"$fdir/forecast_$d$suffix.parquet\" $extra"
}

# Night forecast for the NEXT session (DSM mtap-night, 17:25 PT = 20:25 ET). The claim file in
# STATE/forecast makes it run once across hosts; forecast-night refuses before 20:20 ET.
night() {
    data_ready || { log "ERROR: $MTAP_DATA_DIR has no .mtap-data-root; not running"; return 1; }
    local nlog d; nlog="$LOGS/night-$(date +%F).log"
    d="$(py python -c "from sourcing_py import forecast_night as f; from datetime import date; print(f.next_session(date.today()))" 2>>"$nlog" | tail -1)"
    [ -n "$d" ] || { log "night: no next session found"; return 1; }
    log "night: forecast-night --date $d (log $nlog)"
    py forecast-night --date "$d" >> "$nlog" 2>&1 || { log "night: FAILED (see $nlog)"; return 1; }
    log "night: $(tail -1 "$nlog")"
}

# The morning (DSM mtap-morning, Mon-Fri 06:14 PT = 09:14 ET). It holds no jobs lock (it waits
# for the clock and must not block the 09:30 start) and a pid file keeps a second one out.
morning() {
    data_ready || { log "ERROR: $MTAP_DATA_DIR has no .mtap-data-root; not running"; return 1; }
    local pidf="$MTAP_STATE_DIR/morning.pid" d; d="$(TZ=America/New_York date +%F)"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
        log "morning: already running (pid $(cat "$pidf")); exiting"; return 0
    fi
    echo $$ > "$pidf"; trap 'rm -f "$pidf"' EXIT
    MLOG="$LOGS/morning-$d.log"
    local short="premarket_live/shortlist-$d.csv" ocr=/data/out
    # 1. the night forecast, if no host has finished it
    step_box 5400 "forecast-night (skipped when a done claim exists)" py forecast-night --date "$d" || true
    # 2. the pre-check at 09:29:30
    wait_et 09:29:30 || { log "morning: more than 2 h before 09:29:30 ET; exiting"; return 1; }
    step_box 120 "precheck" py etrade premarket precheck --date "$d" || log "morning: no shortlist"
    # 3. prices at 09:30:00, then the open until 09:32
    wait_et 09:30:00
    step_box 120 "prices" py etrade premarket prices --date "$d" --symbols-file "$ocr/$short" || true
    while [ "$(et_secs)" -lt $((9 * 3600 + 32 * 60)) ]; do
        sleep 20
        step_box 60 "prices --open" py etrade premarket prices --date "$d" --symbols-file "$ocr/$short" --open || break
    done
    step_box 600 "export-next (prices)" py etrade export-next --date "$d" --prices "$ocr/premarket_live/prices-$d.jsonl" || true
    step_box 900 "predict px" predict px "$ocr/$short" || true
    step_box 300 "screener-score" screener px "$d" "$ocr/$short" || true
    # 4. the full pre-market at 09:45
    wait_et 09:45:00
    step_box 1200 "premarket full" py etrade premarket full --date "$d" || return 1
    step_box 600 "export-next (full)" py etrade export-next --date "$d" --premarket "$ocr/premarket_live/premarket-$d.jsonl" || true
    step_box 900 "predict full" predict full || true
    step_box 300 "screener-score (full)" screener full "$d" || true
    log "morning: done"
}

# The one-off 5-year inputs load. Every step is resumable and idempotent; a failed step is
# logged and the next one still runs (queue the job again to finish). Run it after the
# evening job (all jobs share one lock).
inputs_backfill() {
    data_ready || { log "ERROR: $MTAP_DATA_DIR has no .mtap-data-root; not running"; return 1; }
    has_key || { log "inputs-backfill: no MASSIVE_API_KEY in $MTAP_SECRETS_FILE; nothing to do"; return 0; }
    local blog edge yday rc=0 src dst step
    blog="$LOGS/inputs-backfill-$(date +%F).log"
    edge="${MTAP_BACKFILL_FROM:-}"
    [ -n "$edge" ] || { log "inputs-backfill: MTAP_BACKFILL_FROM (the first date with minute aggs) is not set in host.env"; return 1; }
    yday="${MTAP_BACKFILL_TO:-$(date -d yesterday +%F)}"
    log "inputs-backfill: from $edge to $yday (log $blog)"
    src="$MTAP_DATA_DIR/out/extended_hours"; dst="$MTAP_DATA_DIR/backup/extended_hours-pre-vwap"
    if [ -d "$dst" ]; then
        echo "backup exists, kept: $dst" >> "$blog"
    elif [ -d "$src" ]; then
        if mkdir -p "$MTAP_DATA_DIR/backup" && cp -a "$src" "$dst.tmp" && mv "$dst.tmp" "$dst"; then
            echo "backed up $src -> $dst" >> "$blog"
        else
            log "inputs-backfill: FAILED to back up extended_hours; stopping"
            return 1
        fi
    fi
    for step in \
        "massive-ext fetch --chunk quarter --parallel ${MTAP_BACKFILL_PARALLEL:-1} --allow-failures 50 --save-bars /data/bars_1m${MTAP_BARS_UNIVERSE:+ --bars-universe $MTAP_BARS_UNIVERSE} --fromdate $edge --todate $yday" \
        "run massive_tickers_cs" \
        "run massive_shares" \
        "run massive_treasury_yields" \
        "run massive_inflation"; do
        echo "$(date '+%F %T') >>> sourcing-py $step" >> "$blog"
        # shellcheck disable=SC2086
        py $step >> "$blog" 2>&1 || { log "inputs-backfill: step failed: $step"; rc=1; }
    done
    log "inputs-backfill: done (rc $rc); see $blog"
    return "$rc"
}

# The Massive feed is 15 min behind, so its window closes after the E*TRADE collector
# stops (16:21 ET). Wait for it to close its last segment, upload and exit, up to 20 min.
wait_massive() {
    if running massive-ws; then
        log "waiting for massive-ws to finish its window and upload (up to 20 min)"
        for _ in $(seq 1 240); do running massive-ws || break; sleep 5; done
        if running massive-ws; then
            log "WARNING: massive-ws still running after 20 min; stopping it"
            compose "${ALL[@]}" stop massive-ws
        else
            log "massive-ws done for the day"
        fi
    fi
    has_role massive-ws || return 0
    py massive-ws status 2>&1 | tee -a "$SUMMARY_LOG"
    # One file per day (safe to interrupt; the next evening finishes it).
    py massive-ws compact 2>&1 | tee -a "$SUMMARY_LOG"
}

evening() {
    log "========== EVENING =========="
    compose "${ALL[@]}" stop etrade
    # SIGTERM: the store closes its open segments and uploads them; then any leftovers.
    compose "${ALL[@]}" stop store
    py intraday upload --drain || log "WARNING: upload --drain failed"
    local left; left="$(find "$MTAP_STATE_DIR/segments" -type f 2>/dev/null | wc -l)"
    if [ "$left" = 0 ]; then log "all segments uploaded"; else log "WARNING: $left segment file(s) still local; they upload on the next run"; fi
    compose "${ALL[@]}" stop redis
    if has_role etrade; then
        py intraday summary --date "$(TZ=America/New_York date +%F)" 2>&1 | tee -a "$SUMMARY_LOG"
    fi
    # Daily bars a morning run missed (never today's: it is final only after 20:00 ET).
    daily
    wait_massive
    find "$LOGS" -name '*.log' -mtime +14 -delete 2>/dev/null
    log "evening done"
}

stop() {
    log "========== STOP =========="
    compose "${ALL[@]}" stop
}

newest() {  # the newest files under DATA/<dir>: time, size, path
    [ -d "$MTAP_DATA_DIR/$1" ] || { echo "  $1: (none)"; return; }
    find "$MTAP_DATA_DIR/$1" -type f ! -path '*@eaDir*' -printf '%TY-%Tm-%Td %TH:%TM  %10s  %P\n' 2>/dev/null \
        | sort | tail -n "${2:-3}" | sed "s#^#  $1: #"
}

status() {
    local out="$LOGS/status.txt" today; today="$(date +%F)"
    {
        echo "=== MTap status $(date '+%F %T %Z') on $(hostname) ==="
        echo "code:  $(cat "$HERE/../../VERSION" 2>/dev/null || echo unknown)"
        echo "data:  $MTAP_DATA_DIR   state: $MTAP_STATE_DIR   roles: ${MTAP_ROLES:-none}"
        echo
        echo "=== containers ==="
        compose "${ALL[@]}" ps -a
        echo
        echo "=== store and collector heartbeats ==="
        py intraday status
        echo
        echo "=== massive-ws ==="
        py massive-ws status
        echo
        echo "=== Redis ==="
        if running redis; then
            echo "  INTRADAY_FEED length $(compose "${ALL[@]}" exec -T redis redis-cli XLEN mtap:evt:INTRADAY_FEED)," \
                 "memory $(compose "${ALL[@]}" exec -T redis redis-cli INFO memory | grep -E '^used_memory_human' | cut -d: -f2 | tr -d '\r')"
        else
            echo "  not running"
        fi
        echo
        echo "=== newest files in $MTAP_DATA_DIR ==="
        newest intraday_feed; newest agg_1s; newest out/daily_summary 2; newest out/extended_hours 2
        echo
        echo "=== local segments not yet uploaded ==="
        echo "  segments $(find "$MTAP_STATE_DIR/segments" -type f 2>/dev/null | wc -l), segments_massive $(find "$MTAP_STATE_DIR/segments_massive" -type f 2>/dev/null | wc -l)"
        echo
        echo "=== errors in today's logs ==="
        grep -iHE 'error|exception|traceback|fatal' "$LOGS"/*-"$today".log 2>/dev/null | tail -n 25 | sed 's#^#  #'
        echo
        echo "=== collector container (its supervisor and restart loop) ==="
        compose "${ALL[@]}" logs --no-color --tail 15 etrade 2>&1
        echo
        echo "=== jobs.log ==="
        tail -n 20 "$LOGS/jobs.log" 2>/dev/null
    } > "$out.tmp" 2>&1
    mv -f "$out.tmp" "$out"
    log "status written to $out"
}

# The heartbeats each role writes about every 10 s while it works, and the ET hours (Mon-Fri)
# they must be fresh: from 20 min after the 06:10 PT start (time for a login) until just
# before the 13:15 PT evening stops them.
WATCHED=(
    "etrade|etrade/collector.heartbeat|0930|1610|E*TRADE collector (polling; stale while it logs in or waits for a code)"
    "etrade|intraday/store.heartbeat|0930|1610|intraday store (Redis to parquet)"
    "massive-ws|massive/collector.heartbeat|0930|1615|massive-ws collector"
)

watch() {
    local state="$MTAP_STATE_DIR/watch.state" alert="$LOGS/alert.txt" stale="${MTAP_WATCH_STALE_MIN:-10}"
    local now dow hm problems="" w role file from to what age before
    now=$(date +%s); dow=$(TZ=America/New_York date +%u); hm=$((10#$(TZ=America/New_York date +%H%M)))
    rm -f "$alert"
    for w in "${WATCHED[@]}"; do
        IFS='|' read -r role file from to what <<< "$w"
        has_role "$role" || continue
        if [ -z "${MTAP_WATCH_ALWAYS:-}" ]; then
            [ "$dow" -le 5 ] && [ "$hm" -ge $((10#$from)) ] && [ "$hm" -lt $((10#$to)) ] || continue
        fi
        if [ -f "$MTAP_STATE_DIR/$file" ]; then
            age=$(( (now - $(stat -c %Y "$MTAP_STATE_DIR/$file")) / 60 ))
            [ "$age" -ge "$stale" ] && problems+="$file: $what, no heartbeat for $age min"$'\n'
        else
            problems+="$file: $what, no heartbeat file"$'\n'
        fi
    done
    before="$(cat "$state" 2>/dev/null)"
    # Only the names decide "new": the ages grow every run.
    if [ -n "$problems" ]; then
        if [ -n "$(comm -13 <(cut -d: -f1 <<< "$before" | sort) <(cut -d: -f1 <<< "$problems" | sed '/^$/d' | sort))" ]; then
            {
                echo "MTap ALERT on $(hostname), $(date '+%F %T %Z'):"
                printf '%s' "$problems" | sed 's/^/  /'
                echo "noVNC (a login waiting for a code): http://${OTP_HOST:-<host>}:${MTAP_NOVNC_PORT:-7900}"
                echo "Status: STATE/logs/status.txt and jobs.log; a fix is often the stop and start jobs."
            } > "$alert"
            printf '%s' "$problems" > "$state"
            log "ALERT: $(printf '%s' "$problems" | cut -d, -f1 | tr '\n' ' ')"
            return 1
        fi
        printf '%s' "$problems" > "$state"
    elif [ -n "$before" ]; then
        rm -f "$state"
        if [ -z "${MTAP_WATCH_ALWAYS:-}" ] && ! { [ "$dow" -le 5 ] && [ "$hm" -ge 930 ] && [ "$hm" -lt 1615 ]; }; then
            return 0   # the day's window closed: not a recovery
        fi
        echo "MTap RECOVERED on $(hostname), $(date '+%F %T %Z'): every heartbeat is fresh again (was: $(cut -d: -f1 <<< "$before" | tr '\n' ' '))" > "$alert"
        log "recovered: every heartbeat is fresh again"
        return 1
    fi
    return 0
}

ws_test() {
    has_key || { log "ws-test: no MASSIVE_API_KEY in $MTAP_SECRETS_FILE"; return 1; }
    if running massive-ws; then log "ws-test: the massive-ws service is running (one connection per key); not testing"; return 1; fi
    log "ws-test: massive-ws follow --no-window --duration 120"
    py massive-ws follow --no-window --duration 120
}

selftest() {
    log "selftest: a py container as $MTAP_UID:$MTAP_GID"
    compose run --rm -T job sh -c '
        id; python -V; ls -ld /app /app/python /app/python/sourcing_py
        ls -la /app/python/sourcing_py | head -8
        cat /app/python/.venv/lib/python3.11/site-packages/*.pth
        python -c "import sys, importlib.util as u; print(sys.path); print(u.find_spec(\"sourcing_py\")); print(u.find_spec(\"sourcing_py.__main__\"))"
        sourcing-py --help | head -3'
    log "selftest: the collector image (secrets as lengths only; name lookups)"
    compose "${ALL[@]}" run --rm --no-deps -T --entrypoint sh etrade -c '
        for v in ET_USERNAME ET_PASSWORD SMS_PHONE_NUMBER SMS_CARRIER SE_VNC_PASSWORD; do
            eval "x=\${$v:-}"; q=no; case "$x" in *\"*|*\'\''*) q=YES ;; esac
            echo "$v: length ${#x}, contains a quote: $q"
        done
        cat /etc/resolv.conf | grep -v "^#"; getent hosts redis || echo "getent: redis not found"
        cd /app/node && node -e "
            const dns = require(\"dns\"), net = require(\"net\");
            dns.lookup(\"redis\", (e, a) => console.log(\"node lookup:\", e ? e.code : a));
            dns.lookup(\"redis\", { hints: dns.ADDRCONFIG }, (e, a) => console.log(\"node lookup ADDRCONFIG:\", e ? e.code : a));
            const s = net.connect(6379, \"redis\", () => { console.log(\"node connect redis:6379 ok\"); s.end(); });
            s.on(\"error\", (e) => console.log(\"node connect:\", e.code));"'
}

queue() {
    local f="$MTAP_STATE_DIR/run-next" next
    [ -f "$f" ] || { QUEUED=0; return 0; }
    next="$(head -n 1 "$f" | tr -d '[:space:]')"
    rm -f "$f"
    case "$next" in
        build|start|daily|evening|stop|status|ws-test|selftest|morning|night|inputs-backfill) JOB="$next"; log "queued job"; dispatch "$next" ;;
        *) log "ignoring queued job '$next' (not one of build start daily evening stop status ws-test selftest morning night inputs-backfill)" ;;
    esac
}

dispatch() {
    case "$1" in
        build) build ;;
        start) start ;;
        daily) daily ;;
        evening) evening ;;
        stop) stop ;;
        status) status ;;
        ws-test) ws_test ;;
        selftest) selftest ;;
        morning) morning ;;
        night) night ;;
        inputs-backfill) inputs_backfill ;;
        watch) watch ;;
        *) return 2 ;;
    esac
}

case "$JOB" in
    build|start|daily|evening|stop|status|ws-test|selftest|morning|night|inputs-backfill|watch|queue|run) ;;
    *) die "usage: $0 {build|start|daily|evening|stop|status|ws-test|selftest|morning|night|inputs-backfill|watch|queue|run ARGS}" ;;
esac
check_secrets
prepare_state
# The watch reads files only, so it runs even while another job holds the lock.
wrc=0
if [ "$JOB" = queue ]; then watch || wrc=$?; fi
# One job at a time; the queue does not wait for a running one (it runs again later).
exec 9> "$MTAP_STATE_DIR/.jobs.lock"
if ! flock -w "$([ "$JOB" = queue ] && echo 0 || echo 1800)" 9; then
    [ "$JOB" = queue ] && exit "$wrc"
    die "another job held the lock for 30 min; giving up"
fi
rc=0 QUEUED=1
case "$JOB" in
    queue) queue || rc=$? ;;
    run) shift; py "$@" || rc=$? ;;
    *) dispatch "$JOB" || rc=$? ;;
esac
# Every job ends with a fresh status.txt, readable without docker (not an empty queue).
case "$JOB" in status|run|watch) ;; queue) [ "$QUEUED" = 1 ] && status ;; *) status ;; esac
fix_owner
[ "$rc" = 0 ] && rc=$wrc
exit "$rc"
