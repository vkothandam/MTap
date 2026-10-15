#!/bin/bash
# mount-nas.sh: mount the Synology share in MTAP_NAS_URL (e.g. smb://user@synology.local/mtap)
# and wait for MTAP_DATA_ROOT to appear.
#
#   ./scripts/storage/mount-nas.sh                   mount (no-op when already mounted)
#   ./scripts/storage/mount-nas.sh --save-password   store the NAS password in the login Keychain
#
# The password comes from the login Keychain (the SMB entry Finder's "Remember this
# password" creates, or --save-password) and is handed to Finder's `mount volume` with the
# user name, so the mount never depends on Finder finding it by itself. Without a Keychain
# entry it falls back to a plain `mount volume`.
# Exit 0 when the data root is reachable, 1 otherwise.

set -u
source "$(dirname "$0")/../schedule/lib.sh"

URL="${MTAP_NAS_URL:-}"
if [ -z "$URL" ]; then
    echo "MTAP_NAS_URL not set; nothing to mount"
    exit 0
fi

# smb://[user@]host/share -> NAS_USER, NAS_HOST, and the URL without the user.
_rest="${URL#smb://}"
case "$_rest" in *@*) NAS_USER="${_rest%%@*}"; _rest="${_rest#*@}" ;; *) NAS_USER="" ;; esac
NAS_HOST="${_rest%%/*}"
PLAIN_URL="smb://$_rest"

if [ "${1:-}" = "--save-password" ]; then
    [ -n "$NAS_USER" ] || { echo "MTAP_NAS_URL has no user (smb://user@host/share)" >&2; exit 1; }
    # -T: let /usr/bin/security read it back from launchd jobs without a Keychain dialog.
    # A trailing -w prompts for the password without echoing it.
    security add-internet-password -U -s "$NAS_HOST" -a "$NAS_USER" -r "smb " \
        -T /usr/bin/security -T /System/Library/CoreServices/NetAuthAgent.app -w \
        && echo "saved the password for $NAS_USER@$NAS_HOST in the login Keychain"
    exit $?
fi

# /Volumes/<share>: the mount point the data root lives under.
volume_of() { echo "$1" | awk -F/ '$2 == "Volumes" && NF >= 3 {print "/" $2 "/" $3}'; }
VOL="$(volume_of "$MTAP_DATA")"
mounted() { [ -n "$VOL" ] && mount | grep -q " on $VOL (" ; }
# macOS has no `timeout`; perl's alarm kills a command that hangs (a stale SMB mount,
# a Keychain prompt nobody answers).
with_timeout() { local s="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }

if mounted; then
    # After sleep an SMB mount can still be listed but hang on every access.
    if with_timeout 20 ls "$VOL" >/dev/null 2>&1; then
        echo "$VOL already mounted"
        exit 0
    fi
    echo "$VOL is mounted but not responding; unmounting it"
    with_timeout 30 diskutil unmount force "$VOL" >/dev/null 2>&1
fi

NAS_PW=""
if [ -n "$NAS_USER" ]; then
    NAS_PW="$(with_timeout 20 security find-internet-password -s "$NAS_HOST" -a "$NAS_USER" -r "smb " -w 2>/dev/null)" || NAS_PW=""
    [ -n "$NAS_PW" ] || echo "no Keychain password for $NAS_USER@$NAS_HOST; trying without one"
fi

# Credentials go through the environment, never argv, so they don't show up in ps.
mount_once() {
    if [ -n "$NAS_PW" ]; then
        MTAP_NAS_PLAIN_URL="$PLAIN_URL" MTAP_NAS_USER="$NAS_USER" MTAP_NAS_PW="$NAS_PW" \
            with_timeout 60 osascript \
            -e 'mount volume (system attribute "MTAP_NAS_PLAIN_URL") as user name (system attribute "MTAP_NAS_USER") with password (system attribute "MTAP_NAS_PW")'
    else
        with_timeout 60 osascript -e "mount volume \"$URL\""
    fi
}

echo "mounting $PLAIN_URL${NAS_USER:+ as $NAS_USER}"
for attempt in 1 2 3; do
    out="$(mount_once 2>&1)" || echo "attempt $attempt: ${out:-mount volume failed}"
    for _ in $(seq 1 15); do
        mounted && { echo "mounted $VOL"; exit 0; }
        [ -z "$VOL" ] && [ -d "$MTAP_DATA" ] && { echo "data root $MTAP_DATA reachable"; exit 0; }
        sleep 2
    done
    echo "attempt $attempt: not mounted yet"
    sleep 10
done
# A leftover empty folder at /Volumes/<share> makes Finder mount at /Volumes/<share>-1,
# where the data root is not.
if [ -n "$VOL" ] && mount | grep -qE " on $VOL-[0-9]+ \("; then
    echo "the share mounted at $(mount | grep -oE " on $VOL-[0-9]+ " | sed 's/^ on //'), not $VOL: remove the empty $VOL folder (sudo rmdir $VOL), unmount, rerun" >&2
    exit 1
fi
echo "could not mount $URL (is the NAS awake? is the password right? rerun with --save-password)" >&2
exit 1
