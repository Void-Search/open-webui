"""Research recovery across replies, branches, users, JSON and split SSE frames."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from open_webui.ravenous_research import conversation, responses, web_tools
from open_webui.ravenous_research.quality import recovery_message

QUESTION = 'Are international chains required, or is any English-speaking hotel suitable?'
STATE = {
    'user_id': 'alice',
    'original_query': 'English-speaking hotels in Bremen',
    'resolved_query': 'English-speaking hotels in Bremen',
    'question': QUESTION,
    'source_domains': ['example.org'],
    'attempted_queries': ['Bremen hotels'],
}


class ChatsFixture:
    def __init__(self):
        self.messages = {
            'answer': {
                'role': 'assistant',
                'done': True,
                'content': QUESTION,
                'meta': {'existing': True, 'ravenous_research': copy.deepcopy(STATE)},
            },
            'reply': {'role': 'user', 'parentId': 'answer', 'content': 'Either is fine'},
            'next': {'role': 'assistant', 'done': False, 'content': '', 'meta': {'existing': True}},
            'other': {'role': 'assistant', 'done': True, 'content': 'Unrelated branch'},
        }

    async def get_chat_by_id(self, chat_id):
        return SimpleNamespace(user_id='alice') if chat_id == 'chat' else None

    async def get_message_by_id_and_message_id(self, chat_id, message_id):
        assert chat_id == 'chat'
        return copy.deepcopy(self.messages.get(message_id))

    async def upsert_message_to_chat_by_id_and_message_id(self, chat_id, message_id, data, **kwargs):
        assert chat_id == 'chat'
        self.messages[message_id].update(copy.deepcopy(data))


def metadata():
    return {'chat_id': 'chat', 'user_message_id': 'reply', 'message_id': 'next'}


def test_reply_survives_reload_and_preserves_constraints_and_metadata():
    chats = ChatsFixture()
    user = SimpleNamespace(id='alice')

    async def resolver(_request, _model, _user, pending, reply):
        assert pending['question'] == QUESTION and reply == 'Either is fine'
        return True, 'Hotels with English-speaking staff in Bremen, including independent hotels'

    async def exercise():
        form = {'model': 'fixture', 'metadata': metadata(), 'messages': [{'role': 'user', 'content': 'Either is fine'}]}
        context = await conversation.prepare_context(None, form, user, chats=chats, resolver=resolver)
        assert context['source_domains'] == ['example.org']
        assert 'Bremen' in context['query'] and 'independent' in context['query']
        result = {
            'sufficient': False,
            'clarification_question': 'Which part of Bremen?',
            'attempts': [{'query': 'Bremen English hotel official'}],
        }
        await conversation.remember_result(form['metadata'], user, result, chats=chats)
        assert chats.messages['next']['meta']['existing'] is True
        assert await conversation.pending_state(metadata(), SimpleNamespace(id='bob'), chats) is None
        chats.messages['next'].update(done=True, content='Which part of Bremen?')
        chats.messages['reply']['parentId'] = 'next'
        reloaded = await conversation.pending_state(metadata(), user, chats)
        assert reloaded['original_query'] == STATE['original_query']
        assert reloaded['attempted_queries'] == ['Bremen English hotel official']

    asyncio.run(exercise())


@pytest.mark.parametrize('change', ['branch', 'cancelled', 'unfinished', 'different_user'])
def test_pending_context_does_not_cross_branch_or_undelivered_question(change):
    chats = ChatsFixture()
    if change == 'branch':
        chats.messages['reply']['parentId'] = 'other'
    elif change == 'cancelled':
        chats.messages['answer']['content'] = 'Stopped'
    elif change == 'unfinished':
        chats.messages['answer']['done'] = False
    else:
        chats.messages['answer']['meta']['ravenous_research']['user_id'] = 'bob'
    assert asyncio.run(conversation.pending_state(metadata(), SimpleNamespace(id='alice'), chats)) is None


def test_new_topic_and_cancel_do_not_continue_previous_search():
    async def resolver(*args):
        return False, 'Explain Python list comprehensions'

    async def forbidden(*args):
        raise AssertionError('Cancellation must not call the model')

    for reply, resolve in [('Explain Python list comprehensions', resolver), ('Stop searching', forbidden)]:
        form = {'model': 'fixture', 'metadata': metadata(), 'messages': [{'role': 'user', 'content': reply}]}
        context = asyncio.run(
            conversation.prepare_context(
                None, form, SimpleNamespace(id='alice'), chats=ChatsFixture(), resolver=resolve
            )
        )
        assert context['source_domains'] == []
        assert 'Bremen' not in context['query']
        assert context['cancelled'] == (reply == 'Stop searching')


def test_lossy_rewrite_cannot_drop_the_users_clarification():
    async def resolver(*args):
        return True, STATE['original_query']

    reply = 'English-speaking staff, please.'
    form = {'model': 'fixture', 'metadata': metadata(), 'messages': [{'role': 'user', 'content': reply}]}
    context = asyncio.run(
        conversation.prepare_context(None, form, SimpleNamespace(id='alice'), chats=ChatsFixture(), resolver=resolver)
    )
    assert STATE['original_query'] in context['query']
    assert reply in context['query']
    assert context['source_domains'] == ['example.org']


def test_partial_context_excludes_unverified_pages_and_invented_passages():
    evidence = {
        'source': {'url': 'https://example.org/page'},
        'text': 'Bremen hotel has an English website.',
        'content_sha256': 'a' * 64,
        'fetched_at': '2026-09-19T00:00:00Z',
    }
    result = {'sufficient': False, 'evidence': [evidence]}
    assert web_tools.evidence_documents(result) == []
    result['supporting_passages'] = [{'url': evidence['source']['url'], 'text': 'Invented staff languages'}]
    assert web_tools.evidence_documents(result) == []
    result['supporting_passages'][0]['text'] = evidence['text']
    assert web_tools.evidence_documents(result)[0]['content'] == evidence['text']


@pytest.mark.parametrize(
    'code, phrase',
    [
        ('no_results', "haven't found"),
        ('fetch_http_403', "couldn't read"),
        ('fetch_timeout', 'ran out of time'),
        ('discovery_unavailable', 'temporarily unavailable'),
    ],
)
def test_failure_reason_produces_one_appropriate_question(code, phrase):
    text = recovery_message({'sufficient': False, 'attempts': [{'failure_codes': [code]}]})
    assert phrase in text and text.count('?') == 1
    assert 'INCOMPLETE' not in text and 'code comparison' not in text


def test_shared_operation_cannot_reset_budget_but_next_request_can(monkeypatch):
    calls = []

    async def research(*args):
        calls.append(args)
        await asyncio.sleep(0)
        return {'attempts': [{}, {}]}

    monkeypatch.setattr(web_tools, 'research', research)

    async def exercise():
        first = SimpleNamespace(state=SimpleNamespace())
        await asyncio.gather(*(web_tools.shared_research(first, SimpleNamespace(id='alice'), str(i)) for i in range(3)))
        assert len(calls) == 1
        await web_tools.shared_research(
            SimpleNamespace(state=SimpleNamespace()), SimpleNamespace(id='alice'), 'clarified'
        )
        assert len(calls) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize('existing', [False, True])
def test_question_appears_once_in_json_and_split_sse(existing):
    question = 'Which café should I check?'
    meta = {'ravenous_research': {'question': question}}
    answer = 'Supported finding.' + ('\n\n' + question if existing else '')
    data = responses.complete_response(answer, 'fixture')
    json_text = responses.finish_json(data, meta)['choices'][0]['message']['content']
    assert json_text.count(question) == 1

    async def exercise():
        source = responses.complete_response(answer, 'fixture', stream=True)
        raw = ''.join([piece async for piece in source.body_iterator]).encode()

        async def split():
            for byte in raw:
                yield bytes([byte])

        frames = ''.join([piece async for piece in responses.question_stream(split(), meta)])
        output = []
        for frame in frames.strip().split('\n\n'):
            payload = frame.removeprefix('data: ')
            if payload != '[DONE]':
                for choice in json.loads(payload).get('choices', []):
                    output.append(choice.get('delta', {}).get('content', ''))
        assert ''.join(output) == json_text
        assert frames.endswith('data: [DONE]\n\n')

    asyncio.run(exercise())


def test_final_browser_output_keeps_supported_answer_and_appends_question_once():
    output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Supported [1].'}]}]
    meta = {'ravenous_research': {'question': QUESTION}}
    conversation.finish_output(output, meta)
    conversation.finish_output(output, meta)
    assert output[0]['content'][0]['text'] == 'Supported [1].\n\n' + QUESTION


def test_popup_choice_resumes_only_its_parent_without_model_rewrite():
    chats = ChatsFixture()
    reply = 'Retry local sources for the original question.'
    state = chats.messages['answer']['meta']['ravenous_research']
    state['recovery'] = {'question': QUESTION, 'choices': [{'label': 'Retry local', 'reply': reply, 'mode': 'local'}]}
    chats.messages['answer']['content'] = 'Concrete retrieval failure details.'

    async def forbidden(*_args):
        raise AssertionError('A structured choice must not depend on a model rewrite')

    body = {'model': 'fixture', 'metadata': metadata(), 'messages': [{'role': 'user', 'content': reply}]}
    result = asyncio.run(
        conversation.prepare_context(None, body, SimpleNamespace(id='alice'), chats=chats, resolver=forbidden)
    )
    assert result['retrieval_mode'] == 'local'
    assert STATE['original_query'] in result['query'] and reply in result['query']
    assert result['source_domains'] == ['example.org']


def test_joint_failure_notice_appears_once_and_question_stays_in_popup():
    notice = 'Web: 5 queries, 2 pages failed. Local knowledge: completed, 0 sources retrieved.'
    meta = {
        'ravenous_research': {
            'pipeline': 'joint',
            'question': 'How would you like to continue?',
            'response_notice': notice,
        }
    }
    response = responses.complete_response('Supported partial answer [1].', 'fixture')
    result = responses.finish_json(response, meta)
    result = responses.finish_json(result, meta)
    text = result['choices'][0]['message']['content']
    assert text.count(notice) == 1
    assert 'How would you like to continue?' not in text


def test_joint_followup_uses_literal_history_not_the_previous_retry_offer():
    chats = ChatsFixture()
    state = chats.messages['answer']['meta']['ravenous_research']
    state.update(
        pipeline='joint',
        original_query='What should I study to become a teacher in Australia?',
        resolved_query='What should I study to become a teacher in Australia?',
        question='How would you like to continue?',
        recovery={'kind': 'retry', 'question': 'How would you like to continue?', 'choices': []},
        report={'summary': 'Search failed with HTTP 403.', 'sources': [{'kind': 'web', 'selected': True}]},
    )
    chats.messages['answer']['content'] = 'Teaching study pathways. Search failed with HTTP 403.'
    reply = 'I am Australian, tailor to a citizen.'

    async def forbidden(*_args):
        raise AssertionError('Do not resolve a free-form follow-up as an answer to a retry offer')

    body = {
        'model': 'fixture',
        'metadata': metadata(),
        'messages': [
            {'role': 'user', 'content': state['original_query']},
            {'role': 'assistant', 'content': chats.messages['answer']['content']},
            {'role': 'user', 'content': reply},
        ],
    }
    result = asyncio.run(
        conversation.prepare_context(
            None, body, SimpleNamespace(id='alice'), chats=chats, resolver=forbidden, joint=True
        )
    )
    assert result['latest_user_message'] == reply
    assert result['history'] == [
        {'role': 'user', 'content': state['original_query']},
        {'role': 'assistant', 'content': 'Teaching study pathways.'},
    ]
    assert 'clarification_question' not in result
    assert reply in result['query']


@pytest.mark.parametrize('kind', ['local', 'saved', None])
def test_private_assistant_evidence_never_enters_public_query_planning(kind):
    state = {'report': {'sources': [{'kind': 'web', 'selected': True}, {'kind': kind, 'selected': True}]}}
    message = {
        'content': 'Private project details',
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Private project details'}]}],
    }
    assert conversation.public_answer_context(message, state) == ''
    state['report']['sources'] = []
    assert conversation.public_answer_context({'content': 'Unclassified source details'}, state) == ''


def test_public_answer_context_uses_visible_native_output_without_reasoning_or_notices():
    state = {
        'report': {'sources': [{'kind': 'web', 'selected': True}]},
        'response_notice': 'Web: some pages failed.',
    }
    message = {
        'content': 'Stale fallback content',
        'output': [
            {'type': 'reasoning', 'content': [{'type': 'text', 'text': 'Internal reasoning'}]},
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': 'Financial support options. Web: some pages failed.'}],
            },
        ],
    }
    assert conversation.public_answer_context(message, state) == 'Financial support options.'
    message['output'].pop()
    assert conversation.public_answer_context(message, state) == ''
    message['output'].append(
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Answer derived from a private tool.'}]}
    )
    message['output'].append({'type': 'function_call', 'name': 'search_notes'})
    assert conversation.public_answer_context(message, state) == ''


def test_long_public_answer_retains_opening_subject_and_final_options_within_budget():
    state = {'report': {'sources': [{'kind': 'web', 'selected': True}]}}
    answer = 'Current subject. ' + 'Detailed explanation. ' * 300 + 'Final option: connection pooling.'
    result = conversation.public_answer_context({'content': answer}, state)
    assert result.startswith('Current subject.')
    assert result.endswith('Final option: connection pooling.')
    assert len(result) <= 3000


def test_joint_history_ignores_tool_payloads_and_keeps_the_prior_user_with_multimodal_reply():
    messages = [
        {'role': 'system', 'content': 'Hidden system instructions'},
        {'role': 'user', 'content': 'Compare the supported releases'},
        {'role': 'tool', 'content': 'Private document content'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Only the current version'}]},
    ]
    result = conversation.joint_context({}, messages, 'Only the current version', None)
    assert result['history'] == [{'role': 'user', 'content': 'Compare the supported releases'}]
    assert result['latest_user_message'] == 'Only the current version'
