#!/bin/bash
# The one command the NAS runs (DSM Task Scheduler, as root): see nas/README.md.
#
# Lives at <MTap-Data>/nas/massive1s/run.sh on the share (copied there by nas/deploy.sh)
# and works out the data folder from that. Starts one detached container that builds the
# binary if the source changed (cargo cache in massive_1s/build/, so a rebuild is quick)
# and then runs `massive1s fetch` as the NAS user, logging to massive_1s/logs/. Running it
# again after a stop or a reboot resumes; it refuses to start a second copy.
# The share path is /volume2/Backups on this NAS (DSM task: bash /volume2/Backups/MTap-Data/nas/massive1s/run.sh).
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DATA="$(cd "$HERE/../.." && pwd -P)"
ARGS="$DATA/massive_1s/run.args"
die() { echo "massive1s: $*" >&2; exit 2; }

[ -f "$DATA/.mtap-data-root" ] || die "$DATA is not the MTap-Data folder (no .mtap-data-root)"
# Everything this script prints (and docker's errors) also goes to logs/task.log, which
# the Mac can read over the share (`./nas/deploy.sh status`).
mkdir -p "$DATA/massive_1s/logs"
exec > >(tee -a "$DATA/massive_1s/logs/task.log") 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') run.sh in $DATA"
[ "$(id -u)" = 0 ] || die "running as $(id -un), not root: only root can start docker on DSM (Task Scheduler > massive1s > Edit > General > User: root)"
[ -f "$ARGS" ] || die "no $ARGS: run nas/deploy.sh on the Mac first"

# run.args is KEY=VALUE lines written by deploy.sh; only these keys and plain values count.
FROM="" TO="" NAS_USER="" WORKERS=32
while IFS='=' read -r key value || [ -n "$key" ]; do
    [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || continue
    case "$key" in
        FROM) FROM="$value" ;; TO) TO="$value" ;; NAS_USER) NAS_USER="$value" ;; WORKERS) WORKERS="$value" ;;
    esac
done < "$ARGS"
[ -n "$FROM" ] && [ -n "$TO" ] && [ -n "$NAS_USER" ] || die "run.args needs FROM, TO and NAS_USER"

KEYFILE="/var/services/homes/$NAS_USER/.massive1s.env"
[ -f "$KEYFILE" ] || die "no key file $KEYFILE (nas/README.md, one-time setup)"
[ "$(stat -c %a "$KEYFILE")" = 600 ] || die "$KEYFILE must be chmod 600"
uid="$(id -u "$NAS_USER")" gid="$(id -g "$NAS_USER")"

if [ -n "$(docker ps -q --filter name='^massive1s$')" ]; then
    die "already running (docker logs -f massive1s, or ./nas/deploy.sh status on the Mac)"
fi
docker rm massive1s >/dev/null 2>&1 || true  # the previous run's stopped container

mkdir -p "$DATA/massive_1s/build"
chown "$uid:$gid" "$DATA/massive_1s/logs" "$DATA/massive_1s/build"
log="massive_1s/logs/run-$(date +%Y%m%d-%H%M%S)-${FROM}_${TO}.log"

# Not --rm: a container that exits keeps its output for `docker logs massive1s`.
docker run -d --name massive1s --user "$uid:$gid" \
    --env-file "$KEYFILE" \
    -e HOME=/tmp -e CARGO_HOME=/data/massive_1s/build/cargo -e CARGO_TARGET_DIR=/data/massive_1s/build/target \
    -e FROM="$FROM" -e TO="$TO" -e WORKERS="$WORKERS" -e LOG="/data/$log" \
    -v "$DATA:/data" \
    rust:1-alpine sh -c '
        { echo "build (skipped if the source is unchanged)";
          cargo build --release --locked --quiet --manifest-path /data/nas/massive1s/Cargo.toml &&
          exec "$CARGO_TARGET_DIR/release/massive1s" fetch --data /data --from "$FROM" --to "$TO" --workers "$WORKERS"
        } >> "$LOG" 2>&1'

echo "massive1s started: $FROM .. $TO, $WORKERS workers; log $DATA/$log"
sleep 30
if [ -z "$(docker ps -q --filter name='^massive1s$')" ]; then
    echo "massive1s exited within 30 s (status $(docker inspect -f '{{.State.ExitCode}}' massive1s 2>/dev/null)):"
    docker logs massive1s 2>&1 | tail -20
    [ -f "$DATA/$log" ] && tail -20 "$DATA/$log"
    exit 1
fi
