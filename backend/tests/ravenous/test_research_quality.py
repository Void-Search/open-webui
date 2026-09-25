"""Research constraints, failure disclosure and execution evidence regression tests."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from open_webui.ravenous_research.quality import (
    execution_requested,
    execution_response,
    initial_query,
    requested_domains,
)


def test_requested_sites_and_contextual_query():
    question = (
        'Current request: search stackoverflow\n'
        'Earlier user context (resolve references only):\nPython dictionary sorting objects'
    )
    assert requested_domains(question) == ['stackoverflow.com']
    assert requested_domains('Search on docs.example.org for sorting') == ['docs.example.org']
    assert requested_domains('search site:example.org for sorting') == ['example.org']
    assert requested_domains("Don't search stackoverflow") == []
    assert requested_domains('What next?\nEarlier user context: search reddit') == []
    assert (
        initial_query(['search stackoverflow', 'Python dictionary sorting objects'], question)
        == 'Python dictionary sorting objects'
    )


def test_execution_intent_and_truthful_error_status():
    assert execution_requested([{'role': 'user', 'content': 'run the code and compare'}])
    assert not execution_requested([{'role': 'assistant', 'content': 'run the code'}])
    assert not execution_requested(
        [{'role': 'user', 'content': 'run the code'}, {'role': 'user', 'content': 'Do not run code'}]
    )
    result = execution_response('partial output', "AttributeError: 'str' object", '')
    assert result['status'] == 'error'
    assert 'Repair and rerun' in result['verification']
    assert execution_response('measured output', '', '')['status'] == 'success'


def test_native_search_delegates_to_joint_retrieval(monkeypatch):
    from open_webui.ravenous_research import pipeline

    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/utils/middleware.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'chat_web_search_handler')
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    namespace = {}

    async def get(key):
        assert key == 'web.search.engine'
        return 'ravenous'

    async def run(request, body, extra, user):
        assert (request, extra, user) == ('request', 'extra', 'user')
        body['joint_retrieval'] = True
        return body

    namespace['Config'] = SimpleNamespace(get=get)
    monkeypatch.setattr(pipeline, 'run', run)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<handler>', 'exec'), namespace)
    result = asyncio.run(namespace['chat_web_search_handler']('request', {}, 'extra', 'user'))
    assert result['joint_retrieval']


def test_execution_is_required_only_when_an_authorized_tool_is_available():
    from open_webui.ravenous_research.quality import EXECUTION_GUIDANCE, execution_requested

    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/utils/middleware.py').read_text())
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "execution_requested(form_data['messages'])"
    )
    module = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))

    def add(text, messages, **kwargs):
        return [*messages, {'role': 'system', 'content': text}]

    for available in (True, False):
        data = {'messages': [{'role': 'user', 'content': 'run the code and benchmark'}]}
        metadata = {'params': {}}
        namespace = dict(
            execution_requested=execution_requested,
            EXECUTION_GUIDANCE=EXECUTION_GUIDANCE,
            form_data=data,
            metadata=metadata,
            add_or_update_system_message=add,
            tools_dict={'execute_code': {}} if available else {},
        )
        exec(compile(module, '<execution-policy>', 'exec'), namespace)
        assert bool(data.get('tool_choice')) is available
        if available:
            assert data['tool_choice']['function']['name'] == 'execute_code'
        else:
            assert 'unavailable' in data['messages'][-1]['content']
    assert not execution_requested(
        [{'role': 'user', 'content': 'run the code'}, {'role': 'user', 'content': 'Tell me about the moon'}]
    )


@pytest.mark.parametrize('stream', [False, True])
def test_unavailable_research_completes_turn_without_inference(stream):
    from open_webui.ravenous_research.quality import ResearchUnavailable

    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/main.py').read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'process_chat')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    async def payload(*args):
        raise ResearchUnavailable()

    async def context(*args):
        assert args[-2] is None  # no unnecessary follow-up model tasks
        return {'fixture': True}

    async def response(value, ctx):
        assert ctx == {'fixture': True}
        return value

    namespace = dict(
        process_chat_payload=payload,
        ResearchUnavailable=ResearchUnavailable,
        build_chat_response_context=context,
        process_chat_response=response,
        asyncio=asyncio,
    )
    exec(compile(module, '<controlled-research-outcome>', 'exec'), namespace)

    async def exercise():
        result = await namespace['process_chat'](None, {'model': 'fixture', 'stream': stream}, None, {}, None)
        if stream:
            body = ''.join([chunk async for chunk in result.body_iterator])
            assert body.endswith('data: [DONE]\n\n') and 'try again' in body
        else:
            assert result['choices'][0]['finish_reason'] == 'stop'
            assert 'try again' in result['choices'][0]['message']['content']

    asyncio.run(exercise())


def test_modified_json_response_recalculates_content_length():
    from starlette.responses import JSONResponse

    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/utils/middleware.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_response_object')
    namespace = {'JSONResponse': JSONResponse}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<response>', 'exec'),
        namespace,
    )
    result = namespace['build_response_object'](
        JSONResponse({'short': True}), {'answer': 'Longer answer with a question?'}
    )
    assert int(result.headers['content-length']) == len(result.body)
