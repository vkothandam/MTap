# Shared setup for the schedule scripts. Source it; do not run it.
#
# launchd starts jobs with a near-empty environment, so this rebuilds what an
# interactive shell would have: PATH (Homebrew, nvm, uv), the repo .env, and the two
# storage roots. Paths come from this file's location, never from a hard-coded home.

MTAP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export MTAP_REPO
export HOME="${HOME:-$(eval echo "~$(id -un)")}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
if [ -d "$HOME/.nvm/versions/node" ]; then
    _node="$(ls -1 "$HOME/.nvm/versions/node" 2>/dev/null | sort -V | tail -1)"
    [ -n "$_node" ] && export PATH="$HOME/.nvm/versions/node/$_node/bin:$PATH"
fi

# Fill the environment from the repo .env without overriding what is already set.
# Same rules as python common/config.py and node common/dotenv.ts: KEY=VALUE, '#'
# comments, optional `export `, optional quotes, a trailing ' # comment' is dropped.
mtap_load_env() {
    local file="$MTAP_REPO/.env" line key value
    [ -f "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        [ -z "$line" ] && continue
        case "$line" in \#*) continue ;; esac
        line="${line#export }"
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"; key="${key%"${key##*[![:space:]]}"}"
        value="${line#*=}"; value="${value#"${value%%[![:space:]]*}"}"
        if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        else
            value="${value%% \#*}"; value="${value%"${value##*[![:space:]]}"}"
        fi
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [ -n "${!key+x}" ] || export "$key=$value"
    done < "$file"
}
mtap_load_env

_mtap_abs() {  # relative paths resolve from the repo root, like the Python config
    case "$1" in .|./) echo "$MTAP_REPO" ;; /*) echo "$1" ;; "~"/*) echo "$HOME/${1#\~/}" ;; *) echo "$MTAP_REPO/$1" ;; esac
}
MTAP_STATE="$(_mtap_abs "${MTAP_STATE_ROOT:-state}")"
MTAP_DATA="$(_mtap_abs "${MTAP_DATA_ROOT:-.}")"
export MTAP_STATE MTAP_DATA
MTAP_LOGS="$MTAP_STATE/logs"
mkdir -p "$MTAP_LOGS"

# The collector drives a browser from launchd: headless, with puppeteer's own Chrome.
export HEADLESS="${HEADLESS:-true}"
export PUPPETEER_CACHE_DIR="${PUPPETEER_CACHE_DIR:-$HOME/.cache/puppeteer}"

MTAP_SCHEDULE_LOG="$MTAP_LOGS/schedule.log"
mtap_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [$$] $*" | tee -a "$MTAP_SCHEDULE_LOG"; }

# The Python CLI, from the repo's uv project.
sourcing_py() { (cd "$MTAP_REPO/python" && uv run --quiet sourcing-py "$@"); }

mtap_redis_cli() {
    local url="${REDIS_URL:-redis://127.0.0.1:6379}"
    redis-cli -u "$url" "$@"
}
