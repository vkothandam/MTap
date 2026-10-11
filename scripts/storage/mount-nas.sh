#!/bin/bash
# mount-nas.sh: mount the Synology share in MTAP_NAS_URL (e.g. smb://synology.local/mtap)
# and wait for MTAP_DATA_ROOT to appear.
#
# Uses Finder's `mount volume`, which takes the password from the login Keychain. Save it
# once: Finder > Go > Connect to Server, enter the URL, tick "Remember this password".
# Exit 0 when the data root is reachable, 1 otherwise.

set -u
source "$(dirname "$0")/../schedule/lib.sh"

URL="${MTAP_NAS_URL:-}"
if [ -z "$URL" ]; then
    echo "MTAP_NAS_URL not set; nothing to mount"
    exit 0
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

echo "mounting $URL"
for attempt in 1 2 3; do
    with_timeout 60 osascript -e "try" -e "mount volume \"$URL\"" -e "end try" >/dev/null 2>&1
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
echo "could not mount $URL (is the NAS awake and the password in Keychain?)" >&2
exit 1
