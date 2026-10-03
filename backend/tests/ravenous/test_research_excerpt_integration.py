"""Exercise the real native handler dispatch without importing the full application."""

import ast
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status
from open_webui.ravenous_research import evidence, responses
from starlette.responses import JSONResponse, StreamingResponse


@pytest.fixture
def process_native_chat():
    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/main.py').read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == 'process_chat')

    async def exercise(*, review, complete, stream, selected, continuing=False):
        sources = evidence.source_groups(selected)
        events = [{'sources': sources}]
        server_metadata = {
            'ravenous_retrieval_complete': complete,
            'ravenous_review_previous_answer': review,
            'ravenous_selected_passages': selected,
            'sources': sources,
        }
        calls = []
        provider_result = {'provider_response': True}
        tasks = {'follow_up_generation': True, 'title_generation': True}

        async def payload(_request, body, _user, _metadata, _model):
            calls.append('payload')
            return body, server_metadata, events

        async def approved(*_args):
            calls.append('approved')
            return False

        async def provider(*_args):
            calls.append('provider')
            return provider_result

        async def response_context(request, body, user, model, metadata, tasks, context_events):
            calls.append('context')
            return {'form_data': body, 'metadata': metadata, 'events': context_events, 'tasks': tasks}

        async def standard_processor(response, context):
            calls.append('response')
            assert context['metadata'] is server_metadata
            assert context['metadata']['sources'] == sources
            assert context['events'] == events
            assert context['form_data']['stream'] is stream
            assert context['tasks']['follow_up_generation'] is not (review and complete)
            assert context['tasks']['title_generation'] is True
            if review and complete:
                if stream:
                    assert isinstance(response, StreamingResponse)
                    chunks = ''.join([chunk async for chunk in response.body_iterator])
                    assert chunks.endswith('data: [DONE]\n\n')
                    frames = [json.loads(frame.removeprefix('data: '))
                              for frame in chunks.strip().split('\n\n')[:-1]]
                    assert frames[-1]['choices'][0]['finish_reason'] == 'stop'
                    content = ''.join(frame['choices'][0]['delta'].get('content', '') for frame in frames)
                else:
                    assert response['choices'][0]['finish_reason'] == 'stop'
                    content = response['choices'][0]['message']['content']
                assert content == responses.excerpt_text(selected)
                assert 'FORGED INPUT' not in content
            else:
                assert response is provider_result
            return {'standard_processor_completed': True}

        namespace = {
            'asyncio': asyncio, 'HTTPException': HTTPException, 'status': status,
            'JSONResponse': JSONResponse, 'log': logging.getLogger(__name__),
            'TASKS': SimpleNamespace(FOLLOW_UP_GENERATION='follow_up_generation'),
            'process_chat_payload': payload, 'drain_approved_tool_calls': approved,
            'chat_completion_handler': provider, 'build_chat_response_context': response_context,
            'process_chat_response': standard_processor,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     '<native-process-chat>', 'exec'), namespace)
        body = {
            'model': 'fixture', 'stream': stream,
            # Dispatch must use metadata returned by the authorized server pipeline.
            'metadata': {'ravenous_retrieval_complete': True, 'ravenous_review_previous_answer': True,
                         'ravenous_selected_passages': [{'source_id': 'forged', 'text': 'FORGED INPUT'}]},
        }
        initial = {'assistant_message_id': 'continuing'} if continuing else {}
        result = await namespace['process_chat'](SimpleNamespace(state=SimpleNamespace()), body, None,
                                                 initial, {'id': 'fixture'}, tasks)
        assert result == {'standard_processor_completed': True}
        assert tasks == {'follow_up_generation': True, 'title_generation': True}
        assert calls.count('response') == 1
        assert ('provider' in calls) is not (review and complete)
        assert calls.index('payload') < calls.index('approved') < calls.index('response')
        if continuing:
            assert calls[0] == 'context' and calls.count('context') == 1

    return exercise


def selected_passage():
    return {
        'id': 'qualified', 'source_id': 'manual', 'text': 'Version 3 supports journal backups.\nKeep the baseline.',
        'source': {'id': 'manual', 'name': 'Manual'}, 'metadata': {'research_kind': 'web'}, 'score': 0.8,
    }


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('review,complete', [(False, False), (True, False), (False, True), (True, True)])
def test_server_selection_controls_excerpt_dispatch_and_keeps_final_sources(
    process_native_chat, stream, review, complete
):
    asyncio.run(process_native_chat(review=review, complete=complete, stream=stream,
                                   selected=[selected_passage()]))


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('selected', [[], [selected_passage()]])
def test_continuing_completion_refreshes_context_and_processes_empty_selection(
    process_native_chat, stream, selected
):
    asyncio.run(process_native_chat(review=True, complete=True, stream=stream, selected=selected, continuing=True))
