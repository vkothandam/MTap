#!/bin/bash
# bootstrap.sh: set up a fresh checkout on a new Mac (safe to rerun).
#
#   ./scripts/setup/bootstrap.sh
#
# 1. checks node (>= 20), npm, uv and redis-server
# 2. installs the Python (uv sync) and Node (npm ci) dependencies
# 3. installs puppeteer's Chrome for the headless E*TRADE login
# 4. copies .env.example -> .env and settings.example.toml -> settings.toml if missing
# 5. mounts the NAS (if MTAP_NAS_URL is set) and runs `storage init`
#
# It does not install the LaunchAgents or the wake/sleep schedule; run
# scripts/schedule/setup-schedule.sh for that once .env is filled in.

set -u
source "$(dirname "$0")/../schedule/lib.sh"

ok() { echo "[ok] $*"; }
info() { echo "[..] $*"; }
err() { echo "[!!] $*" >&2; }

missing=0
for tool in node npm uv redis-server redis-cli; do
    if command -v "$tool" >/dev/null 2>&1; then ok "$tool: $(command -v "$tool")"; else err "$tool not found"; missing=1; fi
done
if [ $missing -eq 1 ]; then
    err "install the missing tools first: brew install node redis; curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
node_major="$(node -p 'process.versions.node.split(".")[0]')"
[ "$node_major" -ge 20 ] || { err "node $(node --version) is too old; need >= 20"; exit 1; }

info "python deps (uv sync)"
(cd "$MTAP_REPO/python" && uv sync) || { err "uv sync failed"; exit 1; }
info "node deps (npm ci)"
(cd "$MTAP_REPO/node" && npm ci) || { err "npm ci failed"; exit 1; }
info "puppeteer Chrome -> $PUPPETEER_CACHE_DIR"
(cd "$MTAP_REPO/node" && npx --yes puppeteer browsers install chrome) || { err "puppeteer browser install failed"; exit 1; }

created=0
if [ ! -f "$MTAP_REPO/.env" ]; then
    cp "$MTAP_REPO/.env.example" "$MTAP_REPO/.env"; chmod 600 "$MTAP_REPO/.env"
    ok "created .env from .env.example: fill in the credentials and storage roots"
    created=1
fi
if [ ! -f "$MTAP_REPO/shared/config/settings.toml" ]; then
    cp "$MTAP_REPO/shared/config/settings.example.toml" "$MTAP_REPO/shared/config/settings.toml"
    ok "created shared/config/settings.toml from the example: set the Massive api_key"
    created=1
fi
if [ $created -eq 1 ]; then
    echo ""
    echo "Edit .env (and settings.toml), then rerun this script to initialise storage."
    exit 0
fi

if [ -n "${MTAP_NAS_URL:-}" ]; then
    info "mounting $MTAP_NAS_URL"
    "$MTAP_REPO/scripts/storage/mount-nas.sh" || err "NAS mount failed; storage init will refuse an unmounted /Volumes path"
fi
info "storage init"
sourcing_py storage init || { err "storage init failed (see above)"; exit 1; }
sourcing_py storage check

echo ""
echo "Next: ./scripts/schedule/mtap-schedule.sh validate, then ./scripts/schedule/setup-schedule.sh test"
