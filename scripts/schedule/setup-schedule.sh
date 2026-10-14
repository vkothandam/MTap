#!/bin/bash
# setup-schedule.sh: install MTap's LaunchAgents and the Mac's wake/sleep schedule.
#
#   ./scripts/schedule/setup-schedule.sh services          Redis + storage consumer only (no wake/sleep)
#   ./scripts/schedule/setup-schedule.sh test              wake in 2 min, collect from 3 to 6 min, then sleep
#   ./scripts/schedule/setup-schedule.sh prod [--with-daily] [--with-massive]   weekdays: wake 06:15, collect 06:16-13:15, sleep
#   ./scripts/schedule/setup-schedule.sh massive [--remove] only the Massive per-second agent; touches
#                                                           nothing else, so it is safe mid-session
#   ./scripts/schedule/setup-schedule.sh status
#   ./scripts/schedule/setup-schedule.sh stop              unload everything, cancel the power schedule
#
#   --disable-mbin   also unload MBin's com.trading.* agents (their plists are moved to
#                    ~/Library/LaunchAgents/disabled-mbin/, nothing in MBin is touched)
#
# Times are the Mac's local time and assume it is set to Pacific (06:15 PT = 09:15 ET,
# 13:15 PT = 16:15 ET). Agents installed:
#   com.mtap.redis               KeepAlive  redis-server scripts/schedule/redis-mtap.conf
#   com.mtap.store               KeepAlive  mtap-schedule.sh run-store
#   com.mtap.collector.startup   weekdays   mtap-schedule.sh start
#   com.mtap.collector.shutdown  weekdays   mtap-schedule.sh sleep
#   com.mtap.sourcing.daily      weekdays   daily-sourcing.sh (only with --with-daily)
#   com.mtap.massive             KeepAlive  mtap-schedule.sh run-massive (`massive` or --with-massive;
#                                a later `prod` without the flag leaves it as it is):
#                                Massive per-second bars 09:00-16:21 ET (15 min delayed, so the
#                                16:00 close arrives); the 13:15 sleep job waits for it
#
# Ported from MBin's setup-trading.sh; paths come from this script's location.

set -u
source "$(dirname "$0")/lib.sh"

WAKE="06:15"; START_H=6; START_M=16
STOP_H=13; STOP_M=15
SLEEP_BACKSTOP="13:40"          # pmset sleeps the Mac even if the shutdown job failed (after the Massive window)
DAILY_H=6; DAILY_M=40           # the previous session's bars are out after midnight ET

LAUNCH_DIR="$HOME/Library/LaunchAgents"
SCHED="$MTAP_REPO/scripts/schedule/mtap-schedule.sh"
UID_N="$(id -u)"
ALL_LABELS="com.mtap.collector.startup com.mtap.collector.shutdown com.mtap.sourcing.daily com.mtap.massive com.mtap.store com.mtap.redis"

ok() { echo "[ok] $*"; }
info() { echo "[..] $*"; }
err() { echo "[!!] $*" >&2; }

plist_path() { echo "$LAUNCH_DIR/$1.plist"; }

xml_escape() { local s="${1//&/&amp;}"; s="${s//</&lt;}"; echo "${s//>/&gt;}"; }

# write_plist LABEL KIND [CALENDAR_XML] -- ARGS...
#   KIND: keepalive | calendar
write_plist() {
    local label="$1" kind="$2" cal="$3"; shift 3
    [ "${1:-}" = "--" ] && shift
    local args="" a
    for a in "$@"; do args="$args        <string>$(xml_escape "$a")</string>
"; done
    local extra=""
    if [ "$kind" = "keepalive" ]; then
        extra="    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>10</integer>
    <key>ExitTimeOut</key><integer>120</integer>"
    else
        extra="    <key>StartCalendarInterval</key>
$cal
    <key>AbandonProcessGroup</key><true/>"
    fi
    cat > "$(plist_path "$label")" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key>
    <array>
$args    </array>
$extra
    <key>WorkingDirectory</key><string>$MTAP_REPO</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$(xml_escape "$PATH")</string>
        <key>HOME</key><string>$(xml_escape "$HOME")</string>
    </dict>
    <key>StandardOutPath</key><string>$MTAP_LOGS/launchd-$label.log</string>
    <key>StandardErrorPath</key><string>$MTAP_LOGS/launchd-$label.log</string>
</dict>
</plist>
EOF
    plutil -lint "$(plist_path "$label")" >/dev/null || { err "bad plist $(plist_path "$label")"; exit 1; }
}

weekdays_at() {  # weekdays_at HOUR MINUTE -> a StartCalendarInterval array, Mon-Fri
    local d out="    <array>"
    for d in 1 2 3 4 5; do
        out="$out
        <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$1</integer><key>Minute</key><integer>$2</integer></dict>"
    done
    echo "$out
    </array>"
}

once_at() {  # once_at HOUR MINUTE -> a single daily StartCalendarInterval
    echo "    <dict><key>Hour</key><integer>$1</integer><key>Minute</key><integer>$2</integer></dict>"
}

unload() {
    launchctl bootout "gui/$UID_N/$1" 2>/dev/null || launchctl unload "$(plist_path "$1")" 2>/dev/null || true
}

load() {
    launchctl bootstrap "gui/$UID_N" "$(plist_path "$1")" 2>/dev/null || launchctl load "$(plist_path "$1")"
    launchctl print "gui/$UID_N/$1" >/dev/null 2>&1 && ok "loaded $1" || err "could not load $1"
}

# MBin's agents (com.trading.startup / .shutdown), loaded now or waiting in LaunchAgents
# to load at the next login.
mbin_labels() {
    {
        launchctl list | awk '{print $3}' | grep -E '^com\.trading\.' || true
        ls "$LAUNCH_DIR" 2>/dev/null | grep -E '^com\.trading\..*\.plist$' | sed 's/\.plist$//' || true
    } | sort -u
}

check_mbin() {
    local found; found="$(mbin_labels)"
    [ -z "$found" ] && return 0
    if [ "${DISABLE_MBIN:-0}" = "1" ]; then
        mkdir -p "$LAUNCH_DIR/disabled-mbin"
        for label in $found; do
            unload "$label"
            [ -f "$(plist_path "$label")" ] && mv "$(plist_path "$label")" "$LAUNCH_DIR/disabled-mbin/"
            ok "disabled MBin agent $label (plist in $LAUNCH_DIR/disabled-mbin/)"
        done
        return 0
    fi
    err "MBin agents found: $(echo $found). They wake the Mac and scrape the same E*TRADE session."
    err "Rerun with --disable-mbin to unload them, or unload them yourself first."
    exit 1
}

install_services() {
    mkdir -p "$LAUNCH_DIR"
    local redis_bin; redis_bin="$(command -v redis-server || true)"
    if launchctl list | grep -q "homebrew.mxcl.redis"; then
        info "Homebrew's redis service is running; using it instead of com.mtap.redis (check it has no persistence: redis-cli CONFIG GET save)"
    elif [ -z "$redis_bin" ]; then
        err "redis-server not found: brew install redis"; exit 1
    else
        unload com.mtap.redis
        if mtap_redis_cli ping >/dev/null 2>&1; then
            info "something already answers on ${REDIS_URL:-redis://127.0.0.1:6379}; not installing com.mtap.redis"
        else
            write_plist com.mtap.redis keepalive "" -- "$redis_bin" "$MTAP_REPO/scripts/schedule/redis-mtap.conf"
            load com.mtap.redis
        fi
    fi
    unload com.mtap.store
    write_plist com.mtap.store keepalive "" -- /bin/bash "$SCHED" run-store
    load com.mtap.store
}

# The Massive per-second collector: its own process; it shares nothing with the E*TRADE
# collector or Redis. Idle outside its window, so KeepAlive is safe.
install_massive() {
    mkdir -p "$LAUNCH_DIR"
    unload com.mtap.massive
    if [ "${REMOVE:-0}" = "1" ]; then
        rm -f "$(plist_path com.mtap.massive)"
        ok "removed com.mtap.massive"
        return 0
    fi
    [ -n "${MASSIVE_API_KEY:-}" ] || { err "the Massive collector needs MASSIVE_API_KEY in .env"; exit 1; }
    write_plist com.mtap.massive keepalive "" -- /bin/bash "$SCHED" run-massive
    load com.mtap.massive
}

power_settings() {
    info "sleep settings as in MBin (sudo): hibernatemode 25, standby 1, standbydelay 0, powernap 0"
    sudo pmset -a hibernatemode 25 standbydelay 0 standby 1 powernap 0
}

setup_prod() {
    check_mbin
    "$SCHED" validate || { err "fix the validation errors first (see above)"; exit 1; }
    install_services
    [ "${WITH_MASSIVE:-0}" = "1" ] && install_massive
    for label in com.mtap.collector.startup com.mtap.collector.shutdown com.mtap.sourcing.daily; do unload "$label"; done
    write_plist com.mtap.collector.startup calendar "$(weekdays_at $START_H $START_M)" -- /bin/bash "$SCHED" start
    write_plist com.mtap.collector.shutdown calendar "$(weekdays_at $STOP_H $STOP_M)" -- /bin/bash "$SCHED" sleep
    load com.mtap.collector.startup
    load com.mtap.collector.shutdown
    if [ "${WITH_DAILY:-0}" = "1" ]; then
        write_plist com.mtap.sourcing.daily calendar "$(weekdays_at $DAILY_H $DAILY_M)" -- /bin/bash "$MTAP_REPO/scripts/schedule/daily-sourcing.sh"
        load com.mtap.sourcing.daily
    else
        rm -f "$(plist_path com.mtap.sourcing.daily)"
    fi
    info "power schedule (sudo): wake MTWRF $WAKE, sleep MTWRF $SLEEP_BACKSTOP (backstop)"
    sudo pmset repeat cancel
    sudo pmset repeat wake MTWRF "$WAKE:00" sleep MTWRF "$SLEEP_BACKSTOP:00"
    power_settings
    echo ""
    echo "Weekdays: wake $WAKE, collector $(printf '%02d:%02d' $START_H $START_M) to $(printf '%02d:%02d' $STOP_H $STOP_M), then flush to the NAS and sleep."
    [ "${WITH_DAILY:-0}" = "1" ] && echo "Daily sourcing at $(printf '%02d:%02d' $DAILY_H $DAILY_M)."
    launchctl print "gui/$UID_N/com.mtap.massive" >/dev/null 2>&1 && echo "Massive per-second bars 09:00-16:21 ET; the sleep job waits for its last upload (backstop $SLEEP_BACKSTOP)."
    status
}

setup_test() {
    check_mbin
    "$SCHED" validate || { err "fix the validation errors first (see above)"; exit 1; }
    install_services
    local now wake
    now=$(date +%s)
    wake="$(date -r $((now + 120)) '+%m/%d/%Y %H:%M:%S')"
    for label in com.mtap.collector.startup com.mtap.collector.shutdown com.mtap.sourcing.daily; do unload "$label"; done
    write_plist com.mtap.collector.startup calendar "$(once_at "$(date -r $((now + 180)) +%-H)" "$(date -r $((now + 180)) +%-M)")" -- /bin/bash "$SCHED" start
    write_plist com.mtap.collector.shutdown calendar "$(once_at "$(date -r $((now + 360)) +%-H)" "$(date -r $((now + 360)) +%-M)")" -- /bin/bash "$SCHED" sleep
    load com.mtap.collector.startup
    load com.mtap.collector.shutdown
    sudo pmset repeat cancel
    sudo pmset schedule wake "$wake"
    power_settings
    echo ""
    echo "Test timeline:"
    echo "  now       put the Mac to sleep:  pmset sleepnow"
    echo "  $(date -r $((now + 120)) '+%H:%M')     wake"
    echo "  $(date -r $((now + 180)) '+%H:%M')     collector starts (log: $MTAP_LOGS/collector-$(date +%F).log)"
    echo "  $(date -r $((now + 360)) '+%H:%M')     collector stops, store flushes, Mac sleeps"
    echo "Afterwards: wake it, run '$0 status', then '$0 prod'. (The test agents fire daily at those times until replaced.)"
}

status() {
    echo ""
    echo "=== Power schedule ==="
    pmset -g sched
    echo ""
    echo "=== LaunchAgents ==="
    launchctl list | grep -E "com\.(mtap|trading)\." || echo "  (none loaded)"
    echo ""
    "$SCHED" status
}

stop_all() {
    "$SCHED" stop
    for label in $ALL_LABELS; do unload "$label"; rm -f "$(plist_path "$label")"; done
    sudo pmset repeat cancel
    ok "MTap agents removed and the power schedule cancelled"
}

cmd="${1:-}"; shift || true
WITH_DAILY=0; WITH_MASSIVE=0; DISABLE_MBIN=0; REMOVE=0
for a in "$@"; do
    case "$a" in
        --with-daily) WITH_DAILY=1 ;;
        --with-massive) WITH_MASSIVE=1 ;;
        --disable-mbin) DISABLE_MBIN=1 ;;
        --remove) REMOVE=1 ;;
        *) err "unknown option $a"; exit 2 ;;
    esac
done
case "$cmd" in
    services) install_services; status ;;
    test) setup_test ;;
    prod|production) setup_prod ;;
    massive) install_massive; "$SCHED" status | sed -n '/Massive/,/^$/p' ;;
    status) status ;;
    stop) stop_all ;;
    *) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
