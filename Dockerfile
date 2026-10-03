# syntax=docker/dockerfile:1
# Ravenous runs model inference and document/audio processing in separate services.
ARG USE_PERMISSION_HARDENING=false

ARG BUILD_HASH=dev-build
# Override at your own risk - non-root configurations are untested
ARG UID=0
ARG GID=0

######## WebUI frontend ########
FROM --platform=$BUILDPLATFORM node:22-alpine3.20@sha256:2289fb1fba0f4633b08ec47b94a89c7e20b829fc5679f9b7b298eaa2f1ed8b7e AS build
ARG BUILD_HASH
ARG UID
ARG GID

# The upstream frontend exceeds Node's default 4 GiB build heap.
ARG NODE_OPTIONS=--max-old-space-size=8192
ENV NODE_OPTIONS=${NODE_OPTIONS}

WORKDIR /app

# to store git revision in build
RUN apk add --no-cache git

COPY package.json package-lock.json ./
RUN npm ci --force

COPY . .
ENV APP_BUILD_HASH=${BUILD_HASH}
RUN npm run build && find build -type f -name '*.map' -delete

# Prepare backend ownership before the final copy so static assets occupy one layer.
# Group 0 write access lets arbitrary OpenShift UIDs update these assets at startup.
RUN chown -R $UID:$GID /app/backend && \
    chgrp -R 0 /app/backend/open_webui/static && \
    chmod -R g=u /app/backend/open_webui/static

######## WebUI backend ########
FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84 AS base

ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/Void-Search/open-webui" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

ARG USE_PERMISSION_HARDENING
ARG UID
ARG GID

ENV PYTHONUNBUFFERED=1 \
    ENV=prod \
    PORT=8080 \
    USE_SLIM_DOCKER=true \
    USE_OLLAMA_DOCKER=false \
    USE_CUDA_DOCKER=false

## Basis URL Config ##
ENV OLLAMA_BASE_URL="/ollama" \
    OPENAI_API_BASE_URL=""

## API Key and Security Config ##
ENV OPENAI_API_KEY="" \
    WEBUI_SECRET_KEY="" \
    SCARF_NO_ANALYTICS=true \
    DO_NOT_TRACK=true \
    ANONYMIZED_TELEMETRY=false

# Keep the offline splitter cache outside the user data mount.
ENV TIKTOKEN_ENCODING_NAME="cl100k_base" \
    TIKTOKEN_CACHE_DIR="/app/backend/tiktoken-cache"

WORKDIR /app/backend

ENV HOME=/root
# Create user and group if not root
RUN if [ $UID -ne 0 ]; then \
    if [ $GID -ne 0 ]; then \
    addgroup --gid $GID app; \
    fi; \
    adduser --uid $UID --gid $GID --home $HOME --disabled-password --no-create-home app; \
    fi

# Make sure the user has access to the app and root directory
RUN chown -R $UID:$GID /app $HOME

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY --chown=$UID:$GID ./backend/requirements-slim.txt ./

# Set UV_LINK_MODE to copy to prevent 0-byte file corruption in QEMU arm64 cross-builds
ENV UV_LINK_MODE=copy

RUN --mount=from=ghcr.io/astral-sh/uv:0.12.10,source=/uv,target=/bin/uv \
    uv pip install --system -r requirements-slim.txt --no-cache-dir && \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && \
    chmod -R a+rX "$TIKTOKEN_CACHE_DIR" && \
    mkdir -p /app/backend/data && chown -R $UID:$GID /app/backend/data

# Install the versioned ravenous_common shared package supplied through the
# "ravenous_common" named build context (built by `./scripts/stack package
# wheel`; see RAVENOUS.md). This installs a wheel only; it is not the retired
# apply_patch.py mechanism.
COPY --from=ravenous_common / /tmp/ravenous-common-wheel/
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.10,source=/uv,target=/bin/uv \
    uv pip install --system --no-deps /tmp/ravenous-common-wheel/*.whl \
    && pip3 check \
    && rm -rf /tmp/ravenous-common-wheel

# copy built frontend files
COPY --chown=$UID:$GID --from=build /app/build /app/build
COPY --chown=$UID:$GID --from=build /app/CHANGELOG.md /app/CHANGELOG.md
COPY --chown=$UID:$GID --from=build /app/package.json /app/package.json

# copy backend files with the ownership and static permissions prepared above
COPY --from=build /app/backend .

EXPOSE 8080

HEALTHCHECK CMD curl --silent --fail http://localhost:${PORT:-8080}/health | jq -ne 'input.status == true' || exit 1

# Minimal, atomic permission hardening for OpenShift (arbitrary UID):
# - Group 0 owns /app and /root
# - Directories are group-writable and have SGID so new files inherit GID 0
RUN if [ "$USE_PERMISSION_HARDENING" = "true" ]; then \
    set -eux; \
    chgrp -R 0 /app /root || true; \
    chmod -R g+rwX /app /root || true; \
    find /app -type d -exec chmod g+s {} + || true; \
    find /root -type d -exec chmod g+s {} + || true; \
    fi

USER $UID:$GID

ARG BUILD_HASH
ENV WEBUI_BUILD_VERSION=${BUILD_HASH}
ENV DOCKER=true

CMD [ "bash", "start.sh"]
