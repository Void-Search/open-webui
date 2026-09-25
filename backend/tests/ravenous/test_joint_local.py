"""Local candidates retain authenticated pagination and citation provenance."""

import ast
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
from open_webui.ravenous_research import local, transport


def install(monkeypatch, name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)


def test_all_accessible_collections_are_paginated_and_attachments_are_preserved(monkeypatch):
    calls, searched = [], []

    async def groups(user):
        assert user == 'alice'
        return [SimpleNamespace(id='readers')]

    async def page(user, **kwargs):
        assert user == 'alice' and kwargs['filter'] == {'user_id': 'alice', 'group_ids': ['readers']}
        assert kwargs['raise_on_error']
        calls.append(kwargs['skip'])
        length = 100 if kwargs['skip'] == 0 else 2
        return SimpleNamespace(
            items=[
                SimpleNamespace(id=f'collection-{kwargs["skip"] + i}', name=f'Collection {i}', meta={})
                for i in range(length)
            ]
        )

    async def config(*_keys):
        return {}

    async def retrieve(**kwargs):
        assert kwargs['candidate_only'] and kwargs['k_reranker'] == 75
        assert kwargs['reranking_function'] is None and kwargs['r'] == 0
        searched.extend(kwargs['items'])
        return [
            {
                'source': {'id': 'collection-101', 'type': 'collection'},
                'document': ['A retrieved passage'],
                'metadata': [{'file_id': 'manual', 'source': 'https://example.org/manual'}],
            }
        ]

    install(monkeypatch, 'open_webui.models.groups', Groups=SimpleNamespace(get_groups_by_member_id=groups))
    install(monkeypatch, 'open_webui.models.knowledge', Knowledges=SimpleNamespace(search_knowledge_bases=page))
    install(monkeypatch, 'open_webui.models.config', Config=SimpleNamespace(get_many=config))
    install(monkeypatch, 'open_webui.retrieval.utils', get_sources_from_items=retrieve)
    install(
        monkeypatch,
        'open_webui.utils.tools',
        get_attached_knowledge=lambda *_args: [{'id': 'model-file', 'type': 'file'}],
    )

    async def emit(_event):
        pass

    result = asyncio.run(
        local.native_sources(
            None, SimpleNamespace(id='alice'), 'Question', ['one', 'two'], [{'id': 'chat-file', 'type': 'file'}], emit
        )
    )
    assert calls == [0, 100] and len(searched) == 104
    meta = result[0]['metadata'][0]
    assert meta['source'] == 'native:manual' and meta['source_url'] == 'https://example.org/manual'
    assert meta['link'] == '/api/v1/files/manual/content'
    assert meta['retrieval_authority'] == {'id': 'collection-101', 'type': 'collection'}


def test_revision_rechecks_use_session_identity_and_exclude_revoked_sources(monkeypatch):
    monkeypatch.setenv('RAVENOUS_RESEARCH_ENABLED', 'true')
    monkeypatch.setenv('RAVENOUS_WEBUI_RESEARCH_KEY', 'fixture')

    def respond(request):
        doc = request.url.path.rsplit('/', 1)[-1]
        if doc == 'revoked':
            return httpx.Response(403)
        return httpx.Response(200, json={'revision_id': 'new' if doc == 'changed' else 'current'})

    @asynccontextmanager
    async def client(endpoint, key, subject, **kwargs):
        assert key == 'fixture' and subject == 'alice'
        async with httpx.AsyncClient(base_url=endpoint, transport=httpx.MockTransport(respond)) as value:
            yield value

    monkeypatch.setattr(transport, 'knowledge_auth_client', client)
    references = [('allowed', 'current'), ('changed', 'old'), ('revoked', 'current')]
    assert asyncio.run(transport.authorized_revisions(SimpleNamespace(id='alice'), references)) == {
        ('allowed', 'current')
    }


def test_native_collection_revocation_is_rechecked(monkeypatch):
    async def check(identifier, user, permission):
        assert user == 'alice' and permission == 'read'
        return identifier == 'allowed'

    install(monkeypatch, 'open_webui.models.knowledge', Knowledges=SimpleNamespace(check_access_by_user_id=check))
    candidates = [
        {'source_id': identifier, 'metadata': {'retrieval_authority': {'type': 'collection', 'id': identifier}}}
        for identifier in ('allowed', 'revoked')
    ]
    assert asyncio.run(local.authorize_sources(SimpleNamespace(id='alice'), candidates)) == {
        'revoked': 'access_revoked'
    }


def test_saved_citations_reference_exact_document_revision(monkeypatch):
    async def hits(*_args):
        return [
            {
                'source': {'document_id': 'doc', 'revision_id': 'revision', 'title': 'Manual'},
                'text': 'Saved supporting passage.',
                'source_url': 'https://example.org/manual',
            }
        ]

    monkeypatch.setattr(local, 'saved', hits)
    result = asyncio.run(local.research_sources(None, 'Question', ['Query']))
    meta = result[0]['metadata'][0]
    assert meta['source'] == 'research:doc:revision'
    assert meta['link'].endswith('/doc/download?revision_id=revision')


def test_local_candidate_path_cannot_make_unbudgeted_url_fetches():
    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/retrieval/utils.py').read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'get_sources_from_items'
    )
    branch = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "item.get('type') == 'url'"
    )
    probe = ast.AsyncFunctionDef(
        name='probe',
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*branch.body, ast.Return(ast.Name('query_result', ast.Load()))],
        decorator_list=[],
    )
    namespace = {'candidate_only': True, 'failed_retrieval_result': lambda result, reason: {**result, 'reason': reason}}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), '<url-branch>', 'exec'), namespace
    )
    assert asyncio.run(namespace['probe']())['reason'] == 'remote_attachment_requires_web'


def test_native_candidate_merge_preserves_other_sources_before_chunk_quota():
    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/retrieval/utils.py').read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'merge_and_sort_query_results'
    )
    namespace = {'CHUNK_HASH_KEY': 'hash', '_content_hash': lambda text: text}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<merge>', 'exec'), namespace)
    result = namespace['merge_and_sort_query_results'](
        [
            {
                'distances': [[0.9] * 10 + [0.8]],
                'documents': [[f'Long source chunk {i}' for i in range(10)] + ['Another source']],
                'metadatas': [[{'file_id': 'long'}] * 10 + [{'file_id': 'short'}]],
            }
        ],
        k=3,
        candidate_only=True,
    )
    assert [meta['file_id'] for meta in result['metadatas'][0]] == ['long', 'short', 'long']
