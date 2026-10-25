# The Python side: the storage consumer, the Massive per-second collector and the daily
# jobs. Build from the repo root (jobs.sh build does this):
#   docker build -f deploy/container/Dockerfile.py -t mtap-py .
# The repo layout is kept under /app, so the config's repo-root defaults resolve the same
# as on a Mac; the roots themselves come from MTAP_DATA_ROOT / MTAP_STATE_ROOT (compose.yml).
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

# tzdata: the jobs log and name files in the host's zone (TZ, as the iMac's Pacific).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/python/.venv/bin:$PATH

# Dependencies first (cached until uv.lock changes). On Linux torch is the CPU wheel
# (pyproject.toml [tool.uv.sources]).
WORKDIR /app/python
COPY python/pyproject.toml python/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY shared /app/shared
COPY schemas /app/schemas
COPY python /app/python
COPY deploy/container/bin /app/deploy/container/bin
# COPY keeps the build context's modes, which depend on the host's filesystem (on the NAS's
# ACL share some folders arrived as 700), so make the code readable by the runtime user.
RUN find /app/shared /app/schemas /app/python /app/deploy -path /app/python/.venv -prune \
        -o -exec chmod u+rwX,go+rX,go-w {} + \
    && uv sync --frozen --no-dev \
    && python -m compileall -q /app/python/sourcing_py \
    && chmod 755 /app/deploy/container/bin/*.sh

# Containers run as the host's data owner (compose.yml `user:`), which has no home here.
ENV HOME=/tmp
CMD ["sourcing-py"]
