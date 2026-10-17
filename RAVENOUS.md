# Ravenous application customisations

This fork owns Open WebUI adapters, research and memory integration, and final
union reranking. Input preparation, protected terms, grammar lifecycle, and context
budgeting live in the versioned `ravenous_common` wheel; fork compatibility exports
and request-context adapters use that implementation.

The parent Ravenous repository owns deployment configuration, dependency packaging,
and backend regression tests. Its [Open WebUI guide](../../../docs/openwebui/README.md)
documents interfaces and runtime settings; its
[deployment guide](../../../docs/common/deployment.md) documents builds, provenance,
rollout, and recovery. Build through `scripts/stack` to supply the common wheel and
record the deployment handoff.

The frontend build defaults to an 8 GiB Node heap, configurable with the
`NODE_OPTIONS` build argument; reserve additional memory for other build processes.

Run focused checks from the parent repository root:

```sh
./tests/check-fork.sh
./tests/browser.sh --project openwebui
npm --prefix services/openwebui/upstream run test:frontend -- --run src/lib/components/chat/Messages
node --test services/openwebui/upstream/scripts/tests/directory-picker.test.mjs
```

Backend test dependencies are listed in `tests/openwebui/ravenous/requirements.txt`
in the parent repository. The parent check commands cover the relocated backend
tests. Keep `package-lock.json` and upstream `uv.lock`
when updating dependencies. The slim Dockerfile builds the frontend and uses
external RAG and LanguageTool services; runtime data remains at `/app/backend/data`.
