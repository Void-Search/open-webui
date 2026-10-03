"""Move only Ravenous-owned retrieval settings to the external model service."""

import json
import os
from pathlib import Path

MODEL_API_KEYS = (
    'rag.embedding_engine',
    'rag.embedding_model',
    'rag.openai.api_base_url',
    'rag.openai.api_key',
    'rag.reranking_engine',
    'rag.reranking_model',
    'rag.external_reranker_url',
    'rag.external_reranker_api_key',
    'rag.external_reranker_timeout',
    'rag.text_splitter',
    'rag.tokenizer_model',
    'rag.tiktoken_encoding_name',
)


async def migrate_model_api_config(config, defaults: dict, data_dir):
    if os.getenv('RAVENOUS_MODEL_API_ENABLED', '').lower() != 'true':
        return

    desired = {key: defaults[key] for key in MODEL_API_KEYS}
    if (
        desired['rag.embedding_engine'] != 'openai'
        or not desired['rag.openai.api_base_url']
        or not desired['rag.openai.api_key']
        or desired['rag.text_splitter'] != 'token'
        or desired['rag.tiktoken_encoding_name'] != 'cl100k_base'
    ):
        raise RuntimeError('Ravenous model API requires an embedding endpoint, key and token splitter')

    current = await config.get_many(*MODEL_API_KEYS)
    if current.get('rag.embedding_model') not in (None, '', desired['rag.embedding_model']):
        raise RuntimeError('Existing WebUI embedding model differs; restore the configured model before startup')
    updates = {key: value for key, value in desired.items() if current.get(key) != value}
    if not updates:
        return

    # Keep the first pre-migration settings for rollback; credentials never reach logs.
    backup = Path(data_dir) / 'ravenous-model-api-config.json'
    try:
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w') as stream:
            json.dump({'config': current}, stream)
    await config.upsert(updates)
