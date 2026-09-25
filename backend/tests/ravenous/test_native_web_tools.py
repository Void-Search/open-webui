"""Native web-search routes must use the mapped backend identity and acquisition."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from open_webui.ravenous_research import web_tools


def test_delegate_identity_and_bounded_search(monkeypatch):
    monkeypatch.setenv('RAVENOUS_RESEARCH_ENABLED', 'true')
    monkeypatch.setenv('RAVENOUS_WEBUI_RESEARCH_KEY', 'fixture-key')
    calls = []

    @asynccontextmanager
    async def client(endpoint, key, subject, **kwargs):
        assert key == 'fixture-key' and subject == 'session-user'

        async def post(path, json):
            calls.append((path, json))
            return httpx.Response(200, request=httpx.Request('POST', endpoint + path), json={'results': []})

        yield SimpleNamespace(post=post)

    monkeypatch.setattr(web_tools, 'knowledge_auth_client', client)
    assert asyncio.run(web_tools.search(SimpleNamespace(id='session-user'), 'query', 100)) == []
    assert calls == [('/v1/tools/search', {'query': 'query', 'limit': 3})]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(web_tools.search(None, 'query'))
    assert exc.value.status_code == 403


@pytest.mark.parametrize('sufficient', [True, False])
def test_process_uses_one_adaptive_operation_and_exposes_gaps(monkeypatch, sufficient):
    calls = []

    async def research(user, query, count, filters, initial, source_domains):
        calls.append((query, filters, initial))
        return {
            'sufficient': sufficient,
            'missing': [] if sufficient else ['Publication date'],
            'attempts': [{'number': 1}, {'number': 2}],
            'supporting_passages': [{'url': 'https://example.org/page', 'text': 'Fetched evidence'}],
            'evidence': [
                {
                    'source': {'url': 'https://example.org/page', 'title': 'Page'},
                    'text': 'Fetched evidence',
                    'fetched_at': '2026-09-19T00:00:00Z',
                    'content_sha256': 'a' * 64,
                }
            ],
        }

    monkeypatch.setattr(web_tools, 'research', research)
    result = asyncio.run(
        web_tools.process(
            SimpleNamespace(id='user'),
            ['query one', 'query two'],
            domain_filters=['example.org'],
            question='Original question',
        )
    )
    assert calls == [('Original question', ['example.org'], 'query one')]
    assert result['loaded_count'] == 1 and result['collection_name'] is None
    assert 'Fetched evidence' in result['docs'][0]['content']
    assert 'INCOMPLETE' not in result['docs'][0]['content']
    assert 'INCOMPLETE' not in result['research_notice']
    assert result['research']['sufficient'] == sufficient


def test_adaptive_adapter_forwards_bounds_and_identity(monkeypatch):
    calls = []

    async def post(user, tool, payload):
        calls.append((user.id, tool, payload))
        return {'evidence': [], 'sufficient': False}

    monkeypatch.setattr(web_tools, '_post', post)
    asyncio.run(web_tools.research(SimpleNamespace(id='user'), 'question', 99))
    assert calls == [('user', 'research', {'query': 'question', 'limit': 3, 'max_attempts': 2, 'domain_filters': []})]


def test_followup_question_keeps_user_context_and_excludes_assistant_claims():
    messages = [
        {'role': 'user', 'content': 'Compare four ways to sort dictionary objects.'},
        {'role': 'assistant', 'content': 'Invented claims should not become the query.'},
        {'role': 'user', 'content': 'Check online for algorithms.'},
    ]
    question = web_tools.contextual_question(messages, messages[-1]['content'])
    assert 'sort dictionary objects' in question
    assert 'Check online for algorithms.' in question
    assert 'Invented' not in question
    assert len(web_tools.contextual_question(messages, 'x' * 5000)) <= 4096


def test_multiple_elliptical_followups_do_not_lose_original_subject():
    messages = [
        {'role': 'user', 'content': 'Suggest ways to improve sleep consistency.'},
        {'role': 'assistant', 'content': 'Unverified assistant assertions stay out of public planning.'},
        {'role': 'user', 'content': 'Yes search for other strategies.'},
        {'role': 'user', 'content': 'Why did that fail?'},
        {'role': 'user', 'content': 'What are the ways I could achieve this?'},
    ]
    question = web_tools.contextual_question(messages, messages[-1]['content'])
    assert 'sleep consistency' in question and 'achieve this' in question
    assert 'Unverified assistant assertions' not in question


@pytest.mark.parametrize(
    'engine,enabled,permitted,mode,expected',
    [
        ('ravenous', True, True, 'native', 1),
        ('ravenous', False, True, 'native', 0),
        ('ravenous', True, False, 'native', 0),
        ('searxng', True, True, 'native', 0),
        ('searxng', True, True, 'legacy', 1),
    ],
)
def test_web_feature_dispatch_searches_before_native_generation(engine, enabled, permitted, mode, expected):
    # Execute the actual middleware feature branch without loading unrelated DB/model services.
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/utils/middleware.py').read_text())
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("'web_search' in features")
    )
    function = ast.AsyncFunctionDef(
        name='dispatch',
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[branch],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    calls = []

    async def get(key):
        return {'web.search.engine': engine, 'web.search.enable': True, 'user.permissions': {}}[key]

    async def permission(*args):
        return permitted

    async def handler(*args):
        calls.append(True)
        return {}

    namespace = dict(
        features={'web_search': enabled},
        Config=SimpleNamespace(get=get),
        user=SimpleNamespace(role='user', id='fixture'),
        has_permission=permission,
        metadata={'params': {'function_calling': mode}},
        request=None,
        extra_params={},
        chat_web_search_handler=handler,
    )
    # form_data is assigned by the branch, so initialize it inside the test function.
    function.body.insert(
        0, ast.Assign(targets=[ast.Name(id='form_data', ctx=ast.Store())], value=ast.Dict(keys=[], values=[]))
    )
    exec(compile(ast.fix_missing_locations(module), '<web-feature-dispatch>', 'exec'), namespace)
    asyncio.run(namespace['dispatch']())
    assert len(calls) == expected
    if engine == 'ravenous' and expected:
        assert namespace['features']['web_search'] is False  # no duplicate native acquisition
