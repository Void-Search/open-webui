# Ravenous application customisations

This fork owns input preparation, protected terms, llama.cpp context budgeting,
and final union reranking. These are direct backend source changes; image builds
do not apply a patch script. Upstream frontend, API version reporting, migrations,
and `/app/backend/data` storage conventions are preserved.

Build from a clean committed checkout:

```sh
revision=$(git rev-parse HEAD)
docker build --build-arg SOURCE_REVISION="$revision" \
  --build-arg BUILD_HASH="$revision" -t "ravenous-openwebui:$revision" .
```

The Dockerfile builds the frontend with `npm ci` and installs the backend's
version-pinned requirements. Keep `package-lock.json` and upstream `uv.lock`
committed when updating dependencies; the Dockerfile follows upstream's
requirements installation, including CPU-specific Torch wheels. Slim builds are
the default: Ravenous prepares model caches in the same image on the application
host. The application image contains no LanguageTool JVM or managed subprocess.

LanguageTool is a separate HTTP service on either the same Docker network or
another host. Set `RAVENOUS_LANGUAGETOOL_BASE_URL` (default
`http://languagetool:8081`) to its container-reachable address. HTTP and HTTPS,
including a reverse-proxy path prefix, are supported. Use a trusted private
network or an HTTPS proxy for communication across machines.

With correction enabled, application startup waits for a successful `/v2/check`
request using `RAVENOUS_INPUT_LANGUAGE` (default `en-AU`).
`RAVENOUS_LANGUAGETOOL_STARTUP_TIMEOUT_SECONDS` defaults to 60. A failed startup
check prevents readiness. Once running, correction retains the configured
request deadline and concurrency limit; service outages leave the original
prompt intact and recovery is automatic on subsequent requests. Disabling
`RAVENOUS_INPUT_CORRECTION_ENABLED` skips all LanguageTool connections.

`RAVENOUS_LLAMA_CPP_BASE_URL` identifies the inference endpoint that supports
the `/props` and `/v1/chat/completions/input_tokens` context-budget APIs. Set it
to the same reachable URL used by the OpenAI provider, including `/v1`, even
when the inference server runs on another host.

Run the application regression tests from the repository root:

```sh
python -m pip install -r backend/tests/ravenous/requirements.txt
PYTHONPATH=backend python -m pytest backend/tests/ravenous
```

These focused tests require pytest, httpx, typer, and uvicorn; full application
dependencies are declared in the upstream project and backend requirements.
Ravenous owns stack integration and RAG evaluation tests. Before updating its
submodule pointer, test and push the new `ravenous` fork commit, then validate
the stack with that commit and the matching actual API version.
