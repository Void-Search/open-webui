"""Research continuations retain intact excerpts without an intent-confidence gate."""

import asyncio
from types import SimpleNamespace

import pytest
from open_webui.ravenous_research import conversation, pipeline


def research_state():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(RERANKING_FUNCTION=None)))
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Which options were missed?'}]}
    return pipeline.NativeResearch(request, body, {'__event_emitter__': None}, None)


@pytest.mark.parametrize(
    'reply,continuation',
    [('Which backup options were missed?', True), ('How much does that backup option cost?', True),
     ('Explain how black holes form.', False)],
)
def test_research_continuations_use_excerpts_without_an_extra_classifier(monkeypatch, reply, continuation):
    research = research_state()
    calls = []
    context = {
        'latest_user_message': reply, 'original_query': reply,
        'previous_original_query': 'Which backup methods are available?',
        'history': [{'role': 'user', 'content': 'Which backup methods are available?'}],
    }

    async def model(_request, _model, _user, instruction, data, **_kwargs):
        calls.append(instruction)
        assert instruction == pipeline.RESOLVE
        assert data['latest_user_message'] == reply
        return {'continuation': continuation, 'resolved_intent': reply}

    monkeypatch.setattr(pipeline, 'model_json', model)
    asyncio.run(research.resolve_followup(context))
    assert calls == [pipeline.RESOLVE]
    assert context['review_previous_answer'] is continuation
    assert research.metadata['ravenous_review_previous_answer'] is continuation
    assert research.report['failures'] == []


@pytest.mark.parametrize('invalid', [False, True])
def test_unresolved_followup_keeps_literal_question_and_safe_excerpts(monkeypatch, invalid):
    research = research_state()
    research.question = 'What was missed?'

    async def unavailable(*_args, **_kwargs):
        if invalid:
            return {'continuation': 'true', 'resolved_intent': 'An untrusted interpretation of the question'}
        raise TimeoutError

    monkeypatch.setattr(pipeline, 'model_json', unavailable)
    context = {'latest_user_message': research.question, 'previous_query': 'Compare backup options'}
    asyncio.run(research.resolve_followup(context))
    assert context['review_previous_answer'] is True
    assert 'continuation' not in context
    assert research.question == 'What was missed?'
    assert research.metadata['ravenous_review_previous_answer'] is True
    assert research.report['failures'] == [{'stage': 'planning', 'code': 'intent_resolution_unavailable'}]


def test_outer_deadline_cannot_restore_unguarded_followup_generation(monkeypatch):
    research = research_state()

    async def pending(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, 'model_json', pending)
    context = {'latest_user_message': 'What was missed?', 'previous_query': 'Compare backup options'}

    async def exercise():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(research.resolve_followup(context), 0.01)

    asyncio.run(exercise())
    assert context['review_previous_answer'] is True
    assert 'continuation' not in context
    assert research.metadata['ravenous_review_previous_answer'] is True


def test_joint_context_rejects_caller_supplied_review_and_retry_metadata():
    injected = {
        'query': 'Caller-controlled query', 'review_previous_answer': True,
        'continuation': True, 'retrieval_mode': 'web',
    }
    body = {
        'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain snapshot backups'}],
        'metadata': {'ravenous_research_context': injected, 'ravenous_review_previous_answer': True},
    }
    research = pipeline.NativeResearch(None, body, {'__event_emitter__': None}, None)
    context = asyncio.run(conversation.prepare_context(None, body, None, joint=True))
    assert context['query'] == 'Explain snapshot backups'
    assert context['review_previous_answer'] is False and 'retrieval_mode' not in context
    assert research.metadata['ravenous_review_previous_answer'] is False


@pytest.mark.parametrize('reply', ['Retry web search', 'How much does this option cost?', 'Cancel'])
def test_saved_review_mode_survives_explicit_retry_only(monkeypatch, reply):
    metadata = {
        'ravenous_review_previous_answer': True,
        'ravenous_research_context': {'original_query': 'Which backup options were missed?'},
    }
    recovery = {
        'kind': 'retry', 'question': 'How would you like to continue?',
        'choices': [{'reply': 'Retry web search', 'mode': 'web'}],
    }
    result = {
        'pipeline': 'joint', 'query': 'Which backup options were missed?', 'sufficient': False,
        'attempts': [], 'report': {'sources': []}, 'recovery': recovery,
        'clarification_question': recovery['question'],
    }
    user = SimpleNamespace(id='alice')
    asyncio.run(conversation.remember_result(metadata, user, result))
    saved = metadata['ravenous_research']
    assert saved['review_previous_answer'] is True

    class Chats:
        async def get_chat_by_id(self, _chat_id):
            return SimpleNamespace(user_id='alice')

        async def get_message_by_id_and_message_id(self, _chat_id, message_id):
            if message_id == 'reply':
                return {'role': 'user', 'content': reply, 'parentId': 'answer'}
            return {'role': 'assistant', 'done': True, 'content': 'Source excerpts',
                    'meta': {'ravenous_research': saved}}

    body = {
        'model': 'model', 'messages': [{'role': 'user', 'content': reply}],
        'metadata': {'chat_id': 'chat', 'user_message_id': 'reply'},
    }
    context = asyncio.run(conversation.prepare_context(None, body, user, chats=Chats(), joint=True))
    assert context['review_previous_answer'] is (reply == 'Retry web search')
    assert context['cancelled'] is (reply == 'Cancel')
    if reply == 'Retry web search':
        assert context['retrieval_mode'] == 'web' and context['continuation'] is True

        async def forbidden(*_args, **_kwargs):
            raise AssertionError('Saved retry intent must not be reclassified')

        monkeypatch.setattr(pipeline, 'model_json', forbidden)
        asyncio.run(research_state().resolve_followup(context))
        assert context['review_previous_answer'] is True


@pytest.mark.parametrize('revoked', [False, True])
def test_review_quotes_restore_original_qualifiers_only_after_source_authorization(monkeypatch, revoked):
    research = research_state()
    original = (
        'Incremental backups are supported. They require a matching baseline. '
        'Restore only with the matching encryption key.'
    )
    research.question = 'Which backup options were missed?'
    research.sources = [{
        'source': {'id': 'manual', 'name': 'Recovery manual'},
        'document': [original], 'metadata': [{'source': 'manual', 'research_kind': 'local'}],
    }]
    research.metadata['ravenous_review_previous_answer'] = True
    research.metadata['ravenous_research_context'] = {'query': research.question}

    async def fit(_request, _body, _user, ordered, _question, **_kwargs):
        return ordered, []

    async def assess(fitted):
        return {
            'sufficient': True, 'missing': [], 'supported_ids': [fitted[0]['id']],
            'supported_text': {fitted[0]['id']: 'Incremental backups are supported.'},
        }

    async def authorize(_user, selected):
        assert selected[0]['text'] == 'Incremental backups are supported.'
        assert selected[0]['original_excerpt'] == original
        return {'manual': 'access_revoked'} if revoked else {}

    monkeypatch.setattr(pipeline.context_budget, 'fit', fit)
    monkeypatch.setattr(research, 'assess', assess)
    monkeypatch.setattr(pipeline.local, 'authorize_sources', authorize)
    asyncio.run(research.evaluate())
    assert research.selected[0]['original_excerpt'] == original
    asyncio.run(research.finish())
    groups = research.metadata['ravenous_retrieval_sources']
    if revoked:
        assert groups == [] and research.selected == []
        assert research.report['answer_basis'] == 'source_limited'
        assert all(message['role'] != 'system' for message in research.body['messages'])
    else:
        assert research.selected[0]['text'] == original
        assert groups[0]['document'] == [original]
        assert original in research.metadata['ravenous_evidence_message']
