"""External model startup migration preserves saved user settings and vector identity."""

import asyncio
import json

import pytest
from open_webui.ravenous_model_api import migrate_model_api_config


class SavedConfig:
    def __init__(self, values):
        self.values = values.copy()
        self.writes = 0

    async def get_many(self, *keys):
        return {key: self.values[key] for key in keys if key in self.values}

    async def upsert(self, updates):
        self.values.update(updates)
        self.writes += 1


@pytest.fixture
def defaults():
    return {
        'rag.embedding_engine': 'openai',
        'rag.embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'rag.openai.api_base_url': 'http://knowledge:8000/v1',
        'rag.openai.api_key': 'fixture-key',
        'rag.reranking_engine': 'external',
        'rag.reranking_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'rag.external_reranker_url': 'http://knowledge:8000/v1/rerank',
        'rag.external_reranker_api_key': 'fixture-key',
        'rag.external_reranker_timeout': '60',
        'rag.text_splitter': 'token',
        'rag.tokenizer_model': '',
        'rag.tiktoken_encoding_name': 'cl100k_base',
    }


def test_migration_preserves_unrelated_settings_and_private_backup(monkeypatch, tmp_path, defaults):
    monkeypatch.setenv('RAVENOUS_MODEL_API_ENABLED', 'true')
    saved = {
        **defaults,
        'rag.embedding_engine': '',
        'rag.reranking_engine': 'sentence_transformers',
        'rag.text_splitter': 'token_transformers',
        'rag.openai.api_key': 'old-fixture-key',
        'rag.chunk_size': 123,
        'user.permissions': {'chat': {'file_upload': False}},
    }
    config = SavedConfig(saved)
    asyncio.run(migrate_model_api_config(config, defaults, tmp_path))
    assert config.values == {**saved, **defaults}
    backup = tmp_path / 'ravenous-model-api-config.json'
    previous = json.loads(backup.read_text())['config']
    assert previous['rag.embedding_engine'] == ''
    assert previous['rag.openai.api_key'] == 'old-fixture-key'
    assert 'user.permissions' not in previous
    assert backup.stat().st_mode & 0o777 == 0o600

    asyncio.run(migrate_model_api_config(config, defaults, tmp_path))
    assert config.writes == 1
    assert json.loads(backup.read_text())['config'] == previous
    defaults['rag.openai.api_key'] = 'rotated-fixture-key'
    asyncio.run(migrate_model_api_config(config, defaults, tmp_path))
    assert config.values['rag.openai.api_key'] == 'rotated-fixture-key'
    assert json.loads(backup.read_text())['config'] == previous


def test_embedding_model_mismatch_fails_before_writes(monkeypatch, tmp_path, defaults):
    monkeypatch.setenv('RAVENOUS_MODEL_API_ENABLED', 'true')
    config = SavedConfig({**defaults, 'rag.embedding_model': 'different-vector-space'})
    with pytest.raises(RuntimeError, match='embedding model differs'):
        asyncio.run(migrate_model_api_config(config, defaults, tmp_path))
    assert config.writes == 0
    assert not list(tmp_path.iterdir())


def test_disabled_integration_leaves_saved_config_alone(monkeypatch, tmp_path, defaults):
    monkeypatch.delenv('RAVENOUS_MODEL_API_ENABLED', raising=False)
    config = SavedConfig({'rag.embedding_engine': ''})
    asyncio.run(migrate_model_api_config(config, defaults, tmp_path))
    assert config.values == {'rag.embedding_engine': ''}
    assert not config.writes
