"""Question-independent synthesis across web and local evidence."""

import ast
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException
from open_webui.ravenous_research import context, evidence, pipeline
from ravenous_common.context import ContextBudgetError


@pytest.fixture(autouse=True)
def actual_message_normalization(monkeypatch):
    # Exercise production pure helpers without importing unrelated DB/config services.
    tree = ast.parse((Path(__file__).parents[2] / 'open_webui/utils/misc.py').read_text())
    names = {'merge_system_messages', 'strip_empty_content_blocks', 'get_content_from_message', 'get_output_text'}
    module = ModuleType('open_webui.utils.misc')
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), '<normalization>', 'exec'),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)


def source(identity, text, kind='local'):
    return {
        'source': {'id': identity, 'name': identity, 'type': 'file'},
        'document': [text],
        'metadata': [{'source': identity, 'research_kind': kind}],
    }


def test_global_reranking_preserves_source_diversity_and_drops_duplicates():
    candidates = evidence.passages(
        [
            source('web', 'First complementary fact. ' * 120, 'web'),
            source('local', 'Second complementary fact from the local manual.'),
            source('duplicate', 'Second complementary fact from the local manual.'),
            source('irrelevant', 'An unrelated event on an unrelated subject.'),
        ]
    )
    seen = []

    def score(query, docs):
        assert query == 'Resolved original intent'
        seen.extend(doc.page_content for doc in docs)
        return [0.95 if 'First' in doc.page_content else 0.8 if 'Second' in doc.page_content else 0.1 for doc in docs]

    ranked, failed = asyncio.run(evidence.rank('Resolved original intent', candidates, score))
    assert not failed and len(seen) > 2
    assert [item['source_id'] for item in ranked[:2]] == ['web', 'local']
    assert not any(item['source_id'] == 'irrelevant' for item in ranked)
    assert any(item['reason'] == 'duplicate' for item in candidates)
    prompt = evidence.context_message(ranked, 'Resolved original intent')['content']
    assert '<source id="1"' in prompt and '<source id="2"' in prompt
    assert 'Second complementary fact' in prompt


def test_reranker_failure_preserves_fair_order_and_requires_verification():
    candidates = evidence.passages(
        [source('one', 'Different sentence. ' * 180), source('two', 'Other source with useful complementary evidence.')]
    )
    ranked, failed = asyncio.run(evidence.rank('Question', candidates, None))
    assert failed and [item['source_id'] for item in ranked[:2]] == ['one', 'two']
    assert all(item['score'] is None for item in ranked)


def test_context_packing_uses_provider_counts_and_never_generates(monkeypatch):
    async def generate(_request, body, _user):
        assert context.probing.get()
        assert body['messages'][0]['role'] == 'system'
        assert sum(message['role'] == 'system' for message in body['messages']) == 1
        count = body['messages'][0]['content'].count('<source id=')
        if count > 2:
            raise ContextBudgetError('local model protected content exceeds context')
        return {'research_context_checked': True, 'payload': body}

    module = ModuleType('open_webui.utils.chat')
    module.generate_chat_completion = generate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    request = SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(MODELS={'model': {}})))
    candidates = evidence.passages([source(str(i), f'Distinct supporting evidence {i}.') for i in range(8)])
    body = {
        'model': 'model',
        'messages': [{'role': 'system', 'content': 'Original instruction'}, {'role': 'user', 'content': 'Question'}],
    }
    selected, messages = asyncio.run(context.fit(request, body, None, candidates, 'Question'))
    assert len(selected) == 2 and messages[0]['content'].count('<source id=') == 2
    assert body['messages'][0]['content'] == 'Original instruction'
    assert not context.probing.get()


@pytest.fixture
def setup(monkeypatch):  # noqa: C901 - One isolated fixture for the independently exercised branches.
    module = ModuleType('open_webui.models.config')

    async def get(_key):
        return None

    module.Config = SimpleNamespace(get=get)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    events, prompts = [], []
    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(
            state=SimpleNamespace(
                RERANKING_FUNCTION=lambda _query, docs: [0.9] * len(docs),
                MODELS={'model': {}},
            )
        ),
    )

    async def emit(event):
        events.append(event)

    async def prepare(_request, body, _user, **_kwargs):
        value = {
            'query': body['messages'][-1]['content'],
            'original_query': body['messages'][-1]['content'],
            'source_domains': [],
            'cancelled': False,
        }
        body['metadata']['ravenous_research_context'] = value
        return value

    async def remember(metadata, _user, result):
        metadata['ravenous_research'] = result

    async def fit(_request, body, _user, ordered, question):
        return ordered, [*body['messages'], evidence.context_message(ordered, question)]

    async def model(_request, _model, _user, instruction, data, **_kwargs):
        prompts.append((instruction, data))
        if instruction == pipeline.PLAN:
            return {'resolved_intent': data['question'], 'queries': [f'variant {i}' for i in range(5)]}
        if instruction == pipeline.VERIFY:
            return {
                'sufficient': True,
                'supported_ids': [item['id'] for item in data['passages']],
                'missing': [],
                'choices': [],
                'clarification_question': None,
            }
        return {'queries': []}

    async def batch(_user, payload, progress):
        assert len(payload['queries']) == 5
        await progress(
            {
                'type': 'query',
                'data': {'id': 'q1', 'query': payload['queries'][0], 'status': 'completed', 'result_count': 1},
            }
        )
        return {
            'calls': 6,
            'pages': [],
            'evidence': [
                {
                    'source': {'url': 'https://example.org/reference', 'title': 'Public reference'},
                    'text': 'The public specification defines the standard interface.',
                    'fetched_at': '2026-09-19T00:00:00Z',
                    'content_sha256': 'a' * 64,
                },
                {
                    'source': {'url': 'https://another.example/limits', 'title': 'Public limits'},
                    'text': 'A second public source explains the boundary conditions.',
                    'fetched_at': '2026-09-19T00:00:00Z',
                    'content_sha256': 'b' * 64,
                },
            ],
        }

    async def native(*_args, **_kwargs):
        return [source('local-manual', 'The private manual describes the installation requirements.')]

    async def saved(*_args):
        return [source('saved-revision', 'The retained revision describes compatibility limits.', 'saved')]

    monkeypatch.setattr(pipeline, 'prepare_context', prepare)
    monkeypatch.setattr(pipeline, 'remember_result', remember)
    monkeypatch.setattr(pipeline, 'model_json', model)
    monkeypatch.setattr(context, 'fit', fit)
    monkeypatch.setattr(pipeline.transport, 'batch', batch)
    monkeypatch.setattr(pipeline.local, 'native_sources', native)
    monkeypatch.setattr(pipeline.local, 'research_sources', saved)
    return request, emit, events, prompts


@pytest.mark.parametrize(
    'question',
    [
        'What explains the observed change?',
        'Suggest interesting activities for a weekend.',
        'How does the protocol handle retries?',
        'Compare the strengths of these two approaches.',
        'What is the current supported release?',
        'What does the saved manual require?',
        'Which edition do you mean?',
        'How does that compare with the earlier version?',
    ],
)
def test_general_pipeline_supplies_web_and_local_evidence_to_generation(setup, question):
    request, emit, events, prompts = setup
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': question}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    prompt = result['messages'][-1]['content']
    assert 'public specification' in prompt and 'private manual' in prompt and 'retained revision' in prompt
    assert prompt.count('<source id=') == 4
    assert len(result['metadata']['ravenous_retrieval_sources']) == 4
    assert all(
        meta['score'] == 0.9
        for source in result['metadata']['ravenous_retrieval_sources']
        for meta in source['metadata']
    )
    assert events[-1]['data']['recovery'] is None
    assert {item['kind'] for item in events[-1]['data']['report']['sources']} == {'web', 'local', 'saved'}
    assert len(next(data for instruction, data in prompts if instruction == pipeline.VERIFY)['passages']) == 4


def test_candidate_limit_is_fair_before_joint_reranking():
    candidates = evidence.passages(
        [
            source('long', ''.join(f'Unique fact {i}. ' for i in range(500))),
            source('short', 'Short independent source with useful evidence.'),
        ]
    )
    bounded = evidence.bound_candidates(candidates, per_kind=3)
    usable = [item for item in bounded if item['reason'] is None]
    assert len(usable) == 3 and any(item['source_id'] == 'short' for item in usable)
    assert any(item['reason'] == 'candidate_limit' for item in bounded)


def test_verification_references_restore_stable_provenance(setup):
    request, emit, _, prompts = setup
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assessment = next(data for instruction, data in prompts if instruction == pipeline.VERIFY)
    assert [item['id'] for item in assessment['passages']] == ['e1', 'e2', 'e3', 'e4']
    assert len(assessment['sources']) == 4
    assert {source['url'] for source in assessment['sources'] if source['kind'] == 'web'} == {
        'https://example.org/reference',
        'https://another.example/limits',
    }
    assert assessment['utc_date'] in result['messages'][-1]['content']
    stable_ids = result['metadata']['ravenous_evidence_assessment']['supported_ids']
    cited_ids = [
        meta['passage_id'] for source in result['metadata']['ravenous_retrieval_sources'] for meta in source['metadata']
    ]
    assert set(stable_ids) == set(cited_ids)
    assert all(identifier.startswith('p') and len(identifier) == 21 for identifier in stable_ids)


@pytest.mark.parametrize('failure', ['timeout', 'unknown_reference'])
def test_failed_verification_never_labels_unchecked_sources_as_supported(setup, monkeypatch, failure):
    request, emit, events, _ = setup
    original = pipeline.model_json

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            assert 20 < kwargs['timeout'] <= 60
            if failure == 'timeout':
                raise TimeoutError
            result['supported_ids'] = ['untrusted-invented-reference']
        return result

    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert result['metadata']['ravenous_retrieval_sources'] == []
    assert 'general knowledge' in result['messages'][-1]['content']
    report = events[-1]['data']['report']
    expected = 'assessment_timeout' if failure == 'timeout' else 'assessment_unavailable'
    assert report['failures'] == [{'stage': 'verification', 'code': expected}]
    assert all(
        not source['selected'] and source['reasons'] == ['verification_incomplete'] for source in report['sources']
    )
    assert events[-1]['data']['recovery']['choices']


def test_partial_and_conflicting_evidence_is_explicit_in_generation(setup, monkeypatch):
    request, emit, events, _ = setup
    original = pipeline.model_json

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            result.update(
                sufficient=False,
                missing=['Current release date is not established.'],
                conflicts=['The public specification and local manual disagree about the limit.'],
            )
        return result

    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Compare the current limits'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    text = result['messages'][-1]['content']
    assert 'partial answer' in text and 'Current release date' in text and 'disagree about the limit' in text
    assert events[-1]['data']['recovery']


def test_revocation_after_retrieval_excludes_content_and_citations(setup, monkeypatch):
    request, emit, events, _ = setup

    async def recheck(_user, _selected):
        return {'local-manual': 'access_revoked'}

    monkeypatch.setattr(pipeline.local, 'authorize_sources', recheck)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert 'private manual' not in result['messages'][-1]['content']
    report = events[-1]['data']['report']
    excluded = next(item for item in report['sources'] if item['id'] == 'local-manual')
    assert not excluded['selected'] and excluded['citation'] is None
    assert excluded['reasons'] == ['access_revoked']


def test_reranker_failure_still_verifies_bounded_candidates(setup):
    request, emit, events, prompts = setup
    request.app.state.RERANKING_FUNCTION = None
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert result['messages'][-1]['content'].count('<source id=') == 4
    assert any(instruction == pipeline.VERIFY for instruction, _ in prompts)
    assert 'joint reranking unavailable' in events[-1]['data']['report']['summary']


def test_cancellation_stops_all_acquisition_branches_without_completing(setup, monkeypatch):
    request, emit, events, _ = setup
    started, stopped = [], []

    async def blocked(*_args, **_kwargs):
        started.append(True)
        try:
            await asyncio.sleep(30)
        finally:
            stopped.append(True)

    monkeypatch.setattr(pipeline.transport, 'batch', blocked)
    monkeypatch.setattr(pipeline.local, 'native_sources', blocked)
    monkeypatch.setattr(pipeline.local, 'research_sources', blocked)

    async def exercise():
        task = asyncio.create_task(
            pipeline.run(
                request,
                {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]},
                {'__event_emitter__': emit},
                SimpleNamespace(id='alice'),
            )
        )
        while len(started) < 3:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(stopped) == 3
        assert not any(event['data']['action'] == 'research_complete' for event in events)

    asyncio.run(exercise())


@pytest.mark.parametrize('fits', [0, 2])
def test_final_context_accounts_for_tools_and_updates_citations(monkeypatch, fits):
    candidates = evidence.passages([source(str(i), f'Distinct supporting evidence {i}.') for i in range(4)])
    original = evidence.context_message(candidates, 'Question')['content']
    meta = {
        'ravenous_selected_passages': candidates,
        'ravenous_evidence_message': original,
        'ravenous_research_context': {'query': 'Question'},
        'ravenous_research': {
            'report': {'sources': evidence.source_report(candidates, candidates), 'summary': 'Retrieval completed.'}
        },
    }
    body = {
        'model': 'model',
        'metadata': meta,
        'tools': [{'type': 'function', 'function': {'name': 'tool'}}],
        'messages': [{'role': 'system', 'content': 'Other instructions\n' + original}],
    }

    async def fit(_request, payload, _user, selected, _question):
        assert payload['tools'] == body['tools']
        assert original not in json.dumps(payload['messages'])
        return selected[:fits], []

    async def remember(metadata, _user, result):
        metadata['ravenous_research'] = result

    monkeypatch.setattr(context, 'fit', fit)
    monkeypatch.setattr(pipeline, 'remember_result', remember)
    monkeypatch.setattr('open_webui.ravenous_research.conversation.remember_result', remember)
    sources = asyncio.run(context.finalize(None, body, None, None))
    assert len(sources) == fits and body['messages'][-1]['content'].count('<source id=') == fits
    if not fits:
        assert 'general knowledge' in body['messages'][-1]['content']
        assert meta['ravenous_research']['answer_basis'] == 'general_knowledge'
    assert not meta['ravenous_research']['sufficient']
    assert all(
        not item['selected'] and item['citation'] is None
        for item in meta['ravenous_research']['report']['sources'][fits:]
    )


def test_web_outage_does_not_prevent_local_answer(setup, monkeypatch):
    request, emit, events, _ = setup

    async def unavailable(*_args):
        raise HTTPException(503, 'unavailable')

    monkeypatch.setattr(pipeline.transport, 'batch', unavailable)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the interface requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert 'private manual' in result['messages'][-1]['content']
    assert events[-1]['data']['report']['local']['native']['status'] == 'completed'
    assert 'service unavailable' in events[-1]['data']['report']['summary']


def test_short_query_plan_is_completed_before_parallel_dispatch(setup, monkeypatch):
    request, emit, events, _ = setup
    original = pipeline.model_json
    plans = []

    async def plan(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.PLAN:
            plans.append(args[4])
            if len(plans) == 1:
                result['queries'] = ['variant 0', 'variant 1', 'variant 1']
        return result

    monkeypatch.setattr(pipeline, 'model_json', plan)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the interface requirements'}]}
    asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert len(plans) == 2 and plans[1]['existing_queries'] == ['variant 0', 'variant 1']
    assert not events[-1]['data']['report']['failures']


def test_recovery_planning_never_receives_private_passages(setup, monkeypatch):
    request, emit, _, prompts = setup
    original = pipeline.model_json

    async def partial(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            result['sufficient'] = False
        return result

    monkeypatch.setattr(pipeline, 'model_json', partial)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the interface requirements'}]}
    asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    recovery = [data for instruction, data in prompts if instruction not in (pipeline.PLAN, pipeline.VERIFY)]
    assert recovery and 'private manual' not in json.dumps(recovery)


def test_total_failure_generates_labeled_general_answer_with_details_and_clickable_retries(setup, monkeypatch):
    request, emit, events, _ = setup

    async def unavailable(*_args, **_kwargs):
        raise HTTPException(503, 'unavailable')

    monkeypatch.setattr(pipeline.transport, 'batch', unavailable)
    monkeypatch.setattr(pipeline.local, 'native_sources', unavailable)
    monkeypatch.setattr(pipeline.local, 'research_sources', unavailable)
    result = asyncio.run(
        pipeline.run(
            request,
            {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirement'}]},
            {'__event_emitter__': emit},
            SimpleNamespace(id='alice'),
        )
    )
    state = result['metadata']['ravenous_research']
    assert 'Local knowledge: failed' in state['response_notice']
    assert state['answer_basis'] == 'general_knowledge'
    assert result['metadata']['ravenous_retrieval_sources'] == []
    assert 'general knowledge' in result['messages'][-1]['content']
    assert len(events[-1]['data']['recovery']['choices']) == 3


def test_engine_failure_details_reach_response_once_while_local_evidence_survives(setup, monkeypatch):
    request, emit, _, _ = setup
    detail = 'brave: rate limited; duckduckgo: connection failed'

    async def failed_search(_user, payload, progress):
        for index, query in enumerate(payload['queries']):
            await progress(
                {
                    'type': 'query',
                    'data': {
                        'id': f'q{index}',
                        'query': query,
                        'status': 'failed',
                        'result_count': 0,
                        'failure_code': 'discovery_unavailable',
                        'failure_detail': detail,
                    },
                }
            )
        return {'calls': 5, 'pages': [], 'evidence': []}

    monkeypatch.setattr(pipeline.transport, 'batch', failed_search)
    result = asyncio.run(
        pipeline.run(
            request,
            {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirement'}]},
            {'__event_emitter__': emit},
            SimpleNamespace(id='alice'),
        )
    )
    state = result['metadata']['ravenous_research']
    assert state['report']['summary'].count(detail) == 1
    assert state['response_notice'].count(detail) == 1
    assert 'Local knowledge: completed' in state['response_notice']
    assert result['metadata']['ravenous_retrieval_sources']


def test_low_scores_receive_semantic_review_and_irrelevant_sources_stay_excluded(setup, monkeypatch):
    request, emit, events, _ = setup
    request.app.state.RERANKING_FUNCTION = lambda _query, docs: [0.001] * len(docs)
    original = pipeline.model_json

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            result['supported_ids'] = [p['id'] for p in args[4]['passages'] if 'private manual' in p['text']]
        return result

    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the local installation'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert len(result['metadata']['ravenous_retrieval_sources']) == 1
    assert 'private manual' in result['messages'][-1]['content']
    rejected = [s for s in events[-1]['data']['report']['sources'] if not s['selected']]
    assert rejected and all(s['reasons'] == ['unsupported'] for s in rejected)


def test_clean_rerank_question_keeps_literal_constraints_in_assessment_and_generation(setup, monkeypatch):
    request, emit, _, prompts = setup
    original = pipeline.model_json
    literal = 'Current request: use version 7 only\nEarlier user context: explain the interface'

    async def plan(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.PLAN:
            result['resolved_intent'] = 'Explain the version 7 interface'
        return result

    def score(question, docs):
        assert question == 'Explain the version 7 interface'
        return [0.9] * len(docs)

    monkeypatch.setattr(pipeline, 'model_json', plan)
    request.app.state.RERANKING_FUNCTION = score
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': literal}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert literal in result['messages'][-1]['content']
    assert not result['metadata']['ravenous_research']['report']['failures']
    assert next(data for instruction, data in prompts if instruction == pipeline.VERIFY)['user_context'] == literal


def test_document_only_failure_does_not_invent_missing_document_facts(setup, monkeypatch):
    request, emit, events, _ = setup

    async def empty(*_args, **_kwargs):
        return []

    monkeypatch.setattr(pipeline.local, 'native_sources', empty)
    body = {
        'model': 'model',
        'messages': [{'role': 'user', 'content': 'Only use the attached files: what was revenue?'}],
    }
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert events[-1]['data']['report']['answer_basis'] == 'source_limited'
    assert 'Do not supply missing document facts' in result['messages'][-1]['content']
    assert result['metadata']['ravenous_retrieval_sources'] == []


def test_planner_failure_still_dispatches_five_queries(setup, monkeypatch):
    request, emit, events, _ = setup
    original = pipeline.model_json

    async def fail_plan(*args, **kwargs):
        if args[3] == pipeline.PLAN:
            raise ValueError('Invalid planner response')
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipeline, 'model_json', fail_plan)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain interface requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert result['metadata']['ravenous_retrieval_sources']
    assert {'stage': 'planning', 'code': 'planning_unavailable'} in events[-1]['data']['report']['failures']


def test_long_fallback_query_variants_remain_distinct():
    assert len(pipeline.complete_queries([], 'x' * 3500)) == 5


@pytest.mark.parametrize(
    'original,reply,intent,continuation',
    [
        (
            'What should I study to become a teacher in Australia?',
            'I am Australian, tailor to a citizen.',
            'What should an Australian citizen study to become a teacher in Australia?',
            True,
        ),
        (
            'Compare deployment options for the project.',
            'Use version 7 and self-hosting only.',
            'Compare self-hosted deployment options for version 7 of the project.',
            True,
        ),
        (
            'Suggest places for a walking holiday.',
            'It needs to be wheelchair accessible.',
            'Suggest wheelchair-accessible places for a walking holiday.',
            True,
        ),
        (
            'How do I become an early learning child educator in South Australia?',
            'What schemes and government support are available for people in South Australia?',
            'What schemes and government support help people become early childhood educators in South Australia?',
            True,
        ),
        (
            'How do I become an early learning child educator in South Australia?',
            'How do black holes form?',
            'How do black holes form?',
            False,
        ),
        (
            'How do I install solar panels in Victoria?',
            'What support is available across all sectors, not just solar?',
            'What government support is available across all sectors in Victoria?',
            False,
        ),
    ],
)
def test_followup_interpretation_precedes_queries_and_cannot_be_overwritten(
    setup, monkeypatch, original, reply, intent, continuation
):
    request, emit, _, prompts = setup
    original_model = pipeline.model_json
    history = [
        {'role': 'user', 'content': original},
        {'role': 'assistant', 'content': 'Options and next steps for: ' + original},
    ]

    async def prepare(_request, body, _user, **kwargs):
        assert kwargs['joint']
        value = {
            'query': original + '\nUser clarification: ' + reply,
            'original_query': reply,
            'latest_user_message': reply,
            'history': history,
            'previous_query': original,
            'previous_original_query': original,
            'previous_outcome': 'Previous retrieval did not verify enough evidence.',
            'source_domains': [],
            'cancelled': False,
        }
        body['metadata']['ravenous_research_context'] = value
        return value

    async def model(*args, **kwargs):
        instruction, data = args[3:5]
        if instruction == pipeline.RESOLVE:
            prompts.append((instruction, data))
            assert data['latest_user_message'] == reply
            assert data['conversation'] == history
            assert 'retrieval did not verify' not in json.dumps(data)
            return {
                'continuation': continuation,
                'resolved_intent': intent,
                'conversation_subject': 'Underlying conversation subject',
            }
        result = await original_model(*args, **kwargs)
        if instruction == pipeline.PLAN:
            assert intent in data['question'] and data['latest_user_message'] == reply
            assert data['conversation'] == history
            assert data['previous_resolved_question'] == original
            assert data['continuation'] is continuation
            assert data['conversation_subject'] == (original if continuation else 'Underlying conversation subject')
            assert 'retrieval did not verify' not in json.dumps(data)
            result['resolved_intent'] = 'A lossy rewrite that drops the new constraint'
            if 'existing_queries' not in data:
                result['queries'] = result['queries'][:4]
        return result

    def score(question, docs):
        assert intent in question
        if continuation:
            assert original in question and reply in question
        else:
            assert question == reply or question == intent
        return [0.9] * len(docs)

    monkeypatch.setattr(pipeline, 'prepare_context', prepare)
    monkeypatch.setattr(pipeline, 'model_json', model)
    request.app.state.RERANKING_FUNCTION = score
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': reply}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert [instruction for instruction, _ in prompts][:2] == [pipeline.RESOLVE, pipeline.PLAN]
    assert len([data for instruction, data in prompts if instruction == pipeline.PLAN]) == 2
    question = result['metadata']['ravenous_research_context']['query']
    assert intent in question
    assert 'A lossy rewrite' not in question
    assert reply in result['messages'][-1]['content']
    assert not result['metadata']['ravenous_research']['report']['failures']


def test_continued_question_keeps_subject_when_the_model_only_repeats_the_latest_turn():
    context = {
        'previous_original_query': 'How do I install rooftop solar in Victoria?',
        'previous_query': 'A previous rewrite with obsolete assumptions.',
        'latest_user_message': 'What government support is available in Victoria?',
    }
    question = pipeline.continued_question(context['latest_user_message'], context)
    assert context['previous_original_query'] in question
    assert context['latest_user_message'] in question
    assert 'obsolete assumptions' not in question
    assert context['conversation_subject'] == context['previous_original_query']
    for _ in range(20):
        context['previous_query'] = question
        question = pipeline.continued_question('What about eligibility?', context)
    assert question.count('Previous subject:') == 1
    context.update(previous_original_query='a' * 4000, latest_user_message='b' * 4000)
    assert len(pipeline.continued_question('c' * 4000, context)) <= 3500
