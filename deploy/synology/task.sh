#!/bin/bash
# task.sh JOB: what DSM Task Scheduler runs (as root) on the NAS; see README.md.
#
#   bash /volume2/Backups/MTap-App/deploy/synology/task.sh start
#
# JOB is any deploy/container/jobs.sh job (build start daily evening stop status ws-test
# selftest queue). Waits for docker after a boot, reads host.env (and host.override.env, written
# by push.sh --trial) next to this file, and appends everything to STATE/logs/jobs.log,
# which the Mac can read over ssh.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export MTAP_ENV_FILE="$HERE/host.env"
export MTAP_ENV_OVERRIDE="$HERE/host.override.env"

# shellcheck disable=SC1090,SC1091
STATE="$(set -a; . "$MTAP_ENV_FILE"; [ -f "$MTAP_ENV_OVERRIDE" ] && . "$MTAP_ENV_OVERRIDE"; echo "$MTAP_STATE_DIR")"
OWNER="$(set -a; . "$MTAP_ENV_FILE"; echo "$MTAP_UID:$MTAP_GID")"
[ -n "$STATE" ] || { echo "task: MTAP_STATE_DIR not set in $MTAP_ENV_FILE" >&2; exit 2; }
mkdir -p "$STATE/logs" && touch "$STATE/logs/jobs.log" && chown "$OWNER" "$STATE" "$STATE/logs" "$STATE/logs/jobs.log"
# Everything goes to jobs.log; fd 3 keeps the task's own output, which DSM mails when the
# task's "send run details only when the script terminates abnormally" is on.
exec 3>&1 >> "$STATE/logs/jobs.log" 2>&1

# On a non-zero exit: the reason (a watch alert) and the end of jobs.log, for the mail.
report() {
    {
        echo "MTap task '$*' on $(hostname) exited $rc at $(date '+%F %T %Z')"
        echo
        [ -s "$STATE/logs/alert.txt" ] && { cat "$STATE/logs/alert.txt"; echo; }
        echo "--- the end of $STATE/logs/jobs.log ---"
        tail -n 30 "$STATE/logs/jobs.log"
    } >&3
}

rc=2
[ "$(id -u)" = 0 ] || { echo "$(date '+%F %T') task: running as $(id -un), not root: only root can use docker on DSM"; report "$@"; exit $rc; }
# Container Manager starts a minute or more after the NAS boots.
for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done
rc=1
docker info >/dev/null 2>&1 || { echo "$(date '+%F %T') task: docker not up after 5 min; not running ${1:-}"; report "$@"; exit $rc; }

bash "$HERE/../container/jobs.sh" "$@"; rc=$?
[ "$rc" = 0 ] || report "$@"
exit "$rc"
