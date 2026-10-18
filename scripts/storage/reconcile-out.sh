#!/bin/bash
# reconcile-out.sh: one-off, on a Mac that kept its data files in the repo's ./out
# before MTAP_DATA_ROOT pointed it at the NAS. Afterwards every reader and writer uses
# DATA_ROOT/out, the one copy both Macs see.
#
#   ./scripts/storage/reconcile-out.sh           report: per dataset, files only here,
#                                                only on the NAS, and on both
#   ./scripts/storage/reconcile-out.sh --apply   copy the local-only files up (a file on
#                                                both is left as the NAS has it), check
#                                                each arrived, then rename ./out to
#                                                ./out.pre-nas (kept until you delete it)
#
# Then reload the bars from the NAS copy, so each date comes from one file:
#   cd python && uv run sourcing-py etrade ingest-eod && uv run sourcing-py etrade derive-features
#
# Needs MTAP_DATA_ROOT and the share mounted (`sourcing-py storage check`).

set -eu
source "$(dirname "$0")/../schedule/lib.sh"

APPLY=0
case "${1:-}" in
    "") ;;
    --apply) APPLY=1 ;;
    *) echo "usage: $0 [--apply]" >&2; exit 2 ;;
esac

LOCAL="$MTAP_REPO/out"
NAS="$MTAP_DATA/out"
if [ -z "${MTAP_DATA_ROOT:-}" ]; then
    echo "MTAP_DATA_ROOT is not set in .env: nothing to reconcile against" >&2
    exit 1
fi
if [ "$(cd "$MTAP_DATA" && pwd -P)" = "$MTAP_REPO" ]; then
    echo "MTAP_DATA_ROOT is the repo root: ./out already is the data root's out/" >&2
    exit 1
fi
if ! sourcing_py storage check; then
    echo "storage not ready: mount the NAS first (scripts/storage/mount-nas.sh)" >&2
    exit 1
fi
if [ ! -d "$LOCAL" ]; then
    echo "no $LOCAL: nothing to reconcile"
    exit 0
fi

printf '%-16s %10s %10s %10s\n' dataset local_only nas_only both
for dir in "$LOCAL"/*/; do
    ds="$(basename "$dir")"
    here="$(cd "$dir" && find . -type f ! -name '.*' | sort)"
    there="$( [ -d "$NAS/$ds" ] && cd "$NAS/$ds" && find . -type f ! -name '.*' | sort || true)"
    lo=$(comm -23 <(echo "$here") <(echo "$there") | grep -c . || true)
    no=$(comm -13 <(echo "$here") <(echo "$there") | grep -c . || true)
    bo=$(comm -12 <(echo "$here") <(echo "$there") | grep -c . || true)
    printf '%-16s %10s %10s %10s\n' "$ds" "$lo" "$no" "$bo"
done

[ $APPLY = 1 ] || { echo; echo "report only; rerun with --apply to copy the local-only files up"; exit 0; }

echo
echo "copying local-only files to $NAS ..."
rsync -a --ignore-existing --exclude '.*' "$LOCAL/" "$NAS/"
missing=$(cd "$LOCAL" && find . -type f ! -name '.*' | while read -r f; do [ -f "$NAS/$f" ] || echo "$f"; done)
if [ -n "$missing" ]; then
    echo "these did not reach the NAS; ./out left in place:" >&2
    echo "$missing" | head -20 >&2
    exit 1
fi
[ -e "$MTAP_REPO/out.pre-nas" ] && { echo "$MTAP_REPO/out.pre-nas already exists; ./out left in place" >&2; exit 1; }
mv "$LOCAL" "$MTAP_REPO/out.pre-nas"
echo "done: every local file is on the NAS; ./out renamed to ./out.pre-nas"
echo "next: cd python && uv run sourcing-py etrade ingest-eod && uv run sourcing-py etrade derive-features"
