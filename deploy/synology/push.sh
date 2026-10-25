#!/bin/bash
# push.sh [--trial]: on the Mac, copy the code to the NAS over ssh; see README.md.
#
#   deploy/synology/push.sh           production: the next jobs use MTap-Data and host.env's roles
#   deploy/synology/push.sh --trial   the trial: MTap-Trial and only the etrade role
#                                     (host.override.env), so no trial row lands in MTap-Data
#   deploy/synology/push.sh status    print the NAS's STATE/logs/status.txt
#
# Copies the core (python/ node/ schemas/ shared/) and deploy/ as a tar stream (DSM refuses
# rsync over ssh unless its rsync service is on), into MTap-App.new, then swaps it in, so a
# push never leaves a half-copied tree. The working tree is copied as it is, uncommitted
# changes included; VERSION records the commit. Nothing secret is copied: no .env, no
# settings.toml, no state. After a push that changes code, run the `build` task.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
NAS="${MTAP_NAS_SSH:-vijayan.synology@192.168.1.100}"
APP=/volume2/Backups/MTap-App
DATA=/volume2/Backups/MTap-Data
TRIAL=/volume2/Backups/MTap-Trial
STATE=/volume2/Backups/MTap-State
ssh_nas() { ssh -o BatchMode=yes "$NAS" "$@"; }

case "${1:-}" in
    "") trial=0 ;;
    --trial) trial=1 ;;
    status) ssh_nas "cat $STATE/logs/status.txt"; exit 0 ;;
    *) echo "usage: $0 [--trial | status]" >&2; exit 2 ;;
esac

cd "$REPO"
dirty="$(git diff --quiet HEAD -- python node schemas shared deploy .dockerignore 2>/dev/null || echo ' +uncommitted changes')"
version="$(git rev-parse --short HEAD)$dirty, pushed $(date '+%F %T') from $(hostname -s)$([ $trial = 1 ] && echo ' (trial)' || true)"

if [ $trial = 1 ]; then
    override='MTAP_DATA_DIR=/volume2/Backups/MTap-Trial\nMTAP_ROLES=etrade\n'
else
    override=''
fi

# The trial data folder: the marker the jobs require, and the common-stock list that
# massive-ext reads.
if [ $trial = 1 ]; then
    ssh_nas "set -e
        mkdir -p $TRIAL/out
        [ -f $TRIAL/.mtap-data-root ] || cp $DATA/.mtap-data-root $TRIAL/.mtap-data-root
        [ -d $TRIAL/out/tickers_cs ] || cp -r $DATA/out/tickers_cs $TRIAL/out/tickers_cs
        echo \"trial data folder: \$(ls $TRIAL $TRIAL/out | tr '\n' ' ')\""
fi

COPYFILE_DISABLE=1 tar -czf - --no-xattrs \
    --exclude '.venv*' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache' \
    --exclude 'node_modules' --exclude 'node/dist' --exclude 'settings.toml' \
    --exclude '.DS_Store' --exclude '*.log' --exclude 'host.override.env' \
    .dockerignore python node schemas shared deploy \
| ssh_nas "set -e
    rm -rf $APP.new && mkdir -p $APP.new
    tar -xzf - -C $APP.new
    printf '%s\n' '$version' > $APP.new/VERSION
    if [ -n '$override' ]; then printf '$override' > $APP.new/deploy/synology/host.override.env; fi
    rm -rf $APP.old
    if [ -d $APP ]; then mv $APP $APP.old; fi
    mv $APP.new $APP && rm -rf $APP.old
    echo \"pushed to $APP: \$(cat $APP/VERSION)\"
    if [ -f $APP/deploy/synology/host.override.env ]; then echo \"override: \$(tr '\n' ' ' < $APP/deploy/synology/host.override.env)\"; else echo 'no override: production (MTap-Data, host.env roles)'; fi"
