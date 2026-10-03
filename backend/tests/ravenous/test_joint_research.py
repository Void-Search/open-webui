"""Question-independent synthesis across web and local evidence."""

import ast
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException
from open_webui.ravenous_research import context, conversation, evidence, pipeline
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
    assert 'Allowed citation markers: [1], [2].' in prompt
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

    async def fit(_request, body, _user, ordered, question, **_kwargs):
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
    assert [len(data['passages']) for instruction, data in prompts if instruction == pipeline.VERIFY] == [2, 4]


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
    assessment = [data for instruction, data in prompts if instruction == pipeline.VERIFY][-1]
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


@pytest.mark.parametrize('failure', ['timeout', 'truncated', 'unknown_reference'])
def test_failed_verification_never_labels_unchecked_sources_as_supported(setup, monkeypatch, failure):
    request, emit, events, _ = setup
    original = pipeline.model_json

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            assert 20 < kwargs['timeout'] <= 60
            if failure == 'timeout':
                raise TimeoutError
            if failure == 'truncated':
                raise pipeline.ModelOutputTruncated
            result['supported_ids'] = ['untrusted-invented-reference']
        return result

    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert result['metadata']['ravenous_retrieval_sources'] == []
    assert 'general knowledge' in result['messages'][-1]['content']
    report = events[-1]['data']['report']
    expected = {'timeout': 'assessment_timeout', 'truncated': 'assessment_truncated'}.get(
        failure, 'assessment_unavailable'
    )
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
    assert result['messages'][-1]['content'].count('<source id=') == 2
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
        while not started:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(stopped) == 1
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


def test_explicit_local_search_does_not_depend_on_web(setup, monkeypatch):
    request, emit, events, _ = setup

    async def unavailable(*_args):
        raise HTTPException(503, 'unavailable')

    monkeypatch.setattr(pipeline.transport, 'batch', unavailable)
    body = {
        'model': 'model',
        'messages': [{'role': 'user', 'content': 'Only search my attached files: explain the interface requirements'}],
    }
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert 'private manual' in result['messages'][-1]['content']
    assert events[-1]['data']['report']['local']['native']['status'] == 'completed'
    assert not events[-1]['data']['report']['queries']


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
            result.update(
                sufficient=False,
                missing=['No independently supported complementary approach.'],
                conflicts=['Public sources disagree about the supported version.'],
            )
        return result

    monkeypatch.setattr(pipeline, 'model_json', partial)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the interface requirements'}]}
    asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    recovery = [data for instruction, data in prompts if instruction not in (pipeline.PLAN, pipeline.VERIFY)]
    assert recovery and 'private manual' not in json.dumps(recovery)
    assert recovery[0]['previous_answer'] == ''
    assert recovery[0]['missing'] == ['No independently supported complementary approach.']
    assert recovery[0]['conflicts'] == ['Public sources disagree about the supported version.']
    assert recovery[0]['utc_date'] == evidence.calendar_context()['utc_date']


def test_prior_private_retrieval_gaps_never_enter_public_verification_or_recovery(setup, monkeypatch):
    request, emit, _, prompts = setup
    original_prepare = pipeline.prepare_context
    original_model = pipeline.model_json
    private_detail = 'PRIVATE SUMMARY: unreleased Project Cedar requires an undisclosed recovery key.'
    public_gap = 'No independently supported complementary approach.'

    async def prepare(*args, **kwargs):
        value = await original_prepare(*args, **kwargs)
        # A preceding mixed-source answer can retain private facts in its evidence
        # gaps even though its visible answer is excluded from public planning.
        value['previous_outcome'] = 'Evidence gaps: ' + private_detail
        return value

    async def model(*args, **kwargs):
        instruction, data = args[3:5]
        assert private_detail not in json.dumps(data)
        result = await original_model(*args, **kwargs)
        if instruction == pipeline.VERIFY:
            result.update(sufficient=False, missing=[public_gap], conflicts=[])
        return result

    monkeypatch.setattr(pipeline, 'prepare_context', prepare)
    monkeypatch.setattr(pipeline, 'model_json', model)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'What other backup approaches are available?'}]}
    asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert any(instruction == pipeline.VERIFY for instruction, _ in prompts)
    recovery = [
        data for instruction, data in prompts
        if instruction not in (pipeline.PLAN, pipeline.RESOLVE, pipeline.VERIFY)
    ]
    assert len(recovery) == 1 and recovery[0]['missing'] == [public_gap]
    assert recovery[0]['previous_answer'] == ''


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
    assert 'Local knowledge: skipped' in state['report']['summary']
    assert state['response_notice'] is None
    assert state['answer_basis'] == 'general_knowledge'
    assert result['metadata']['ravenous_retrieval_sources'] == []
    assert 'general knowledge' in result['messages'][-1]['content']
    assert len(events[-1]['data']['recovery']['choices']) == 3


def test_engine_failure_details_stay_in_report_and_local_search_is_skipped(setup, monkeypatch):
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
    assert state['response_notice'] is None
    assert 'Local knowledge: skipped' in state['report']['summary']
    assert detail not in result['messages'][-1]['content']
    assert not result['metadata']['ravenous_retrieval_sources']


def test_low_scores_receive_semantic_review_and_irrelevant_sources_stay_excluded(setup, monkeypatch):
    request, emit, events, _ = setup
    request.app.state.RERANKING_FUNCTION = lambda _query, docs: [0.001] * len(docs)
    original = pipeline.model_json

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            result['supported_ids'] = [p['id'] for p in args[4]['passages'] if 'public specification' in p['text']]
        return result

    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the local installation'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert len(result['metadata']['ravenous_retrieval_sources']) == 1
    assert 'public specification' in result['messages'][-1]['content']
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
            assert data['utc_date'] == evidence.calendar_context()['utc_date']
            assert 'retrieval did not verify' not in json.dumps(data)
            return {
                'continuation': continuation,
                'resolved_intent': intent,
                'conversation_subject': 'Underlying conversation subject',
            }
        result = await original_model(*args, **kwargs)
        if instruction == pipeline.VERIFY:
            assert data['previous_answer'] == (history[-1]['content'] if continuation else '')
        if instruction == pipeline.PLAN:
            assert data['question'] == intent and data['latest_user_message'] == reply
            assert data['conversation'] == history[:1]
            assert original in data['literal_user_context'] and reply in data['literal_user_context']
            assert history[-1]['content'] not in json.dumps(data)
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
    assert result['metadata']['ravenous_review_previous_answer'] is continuation
    assert len([data for instruction, data in prompts if instruction == pipeline.PLAN]) == 2
    question = result['metadata']['ravenous_research_context']['query']
    assert intent in question
    assert 'A lossy rewrite' not in question
    assert reply in result['messages'][-1]['content']
    assert result['metadata']['ravenous_evidence_assessment']['previous_answer'] == (
        history[-1]['content'] if continuation else ''
    )
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


def test_joint_research_uses_configured_external_reranker(setup, monkeypatch):
    request, emit, events, _ = setup

    async def external_engine(_key):
        return 'external'

    monkeypatch.setattr(sys.modules['open_webui.models.config'].Config, 'get', external_engine)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert all(
        meta['score'] == 0.9
        for item in result['metadata']['ravenous_retrieval_sources']
        for meta in item['metadata']
    )
    assert 'joint reranking unavailable' not in events[-1]['data']['report']['summary']


def test_rejected_saved_sources_cannot_produce_a_clarification(setup, monkeypatch):
    request, emit, events, _ = setup
    original = pipeline.model_json

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            result.update(
                sufficient=False,
                supported_ids=[],
                clarification_question='The sources define content. Can you provide event information?',
                choices=['No event information', 'The sources only define content'],
            )
        return result

    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Bremen weekend events'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    recovery = events[-1]['data']['recovery']
    assert recovery['kind'] == 'retry'
    assert 'define content' not in json.dumps(recovery)
    assert result['metadata']['ravenous_retrieval_sources'] == []


@pytest.mark.parametrize('sufficient', [False, True])
def test_local_search_starts_only_after_sufficient_web_evidence(setup, monkeypatch, sufficient):
    request, emit, events, _ = setup
    original = pipeline.model_json
    verified, calls = [], []

    async def model(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            verified.append(True)
            result['sufficient'] = sufficient
        return result

    async def native(_request, _user, question, queries, *_args, **_kwargs):
        assert verified and sufficient and queries == [question]
        calls.append('native')
        return []

    async def saved(_user, question, queries):
        assert verified and sufficient and queries == [question]
        calls.append('saved')
        return []

    monkeypatch.setattr(pipeline, 'model_json', model)
    monkeypatch.setattr(pipeline.local, 'native_sources', native)
    monkeypatch.setattr(pipeline.local, 'research_sources', saved)
    asyncio.run(pipeline.run(request, {'model': 'model', 'messages': [{'role': 'user', 'content': 'Question'}]},
                             {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    assert set(calls) == ({'native', 'saved'} if sufficient else set())


def test_only_verified_sentences_from_a_mixed_event_passage_reach_the_answer(setup, monkeypatch):
    request, emit, _, _ = setup
    original = pipeline.model_json
    wanted = '[Community concert](https://example.org/concert)\nOctober 4, 2026, Bremen, Germany.\nDoors at 19:00.'

    async def batch(*_args):
        return {'calls': 1, 'pages': [], 'evidence': [{
            'source': {'url': 'https://example.org/events', 'title': 'Events'},
            'text': wanted + '\nPast market on September 29, 2026. Future parade on August 28, 2027.',
            'fetched_at': '2026-09-29T00:00:00Z', 'content_sha256': 'a' * 64,
        }]}

    async def assess(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[3] == pipeline.VERIFY:
            chosen = [p for p in args[4]['passages'] if p['text'] in wanted.splitlines()]
            assert len({p['block'] for p in chosen}) == 1
            result['supported_ids'] = [p['id'] for p in chosen]
        return result

    monkeypatch.setattr(pipeline.transport, 'batch', batch)
    monkeypatch.setattr(pipeline, 'model_json', assess)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Bremen this weekend'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    answer_context = result['messages'][-1]['content']
    assert wanted in answer_context
    assert 'Past market' not in answer_context and 'Future parade' not in answer_context
    assert 'Web:' not in answer_context and 'Local knowledge:' not in answer_context
    assert result['metadata']['ravenous_retrieval_sources'][0]['document'] == [wanted]


def test_weekend_reference_is_stable_during_the_weekend():
    import datetime as dt
    expected = ['2026-10-02', '2026-10-03', '2026-10-04']
    for day in (dt.date(2026, 9, 29), dt.date(2026, 10, 2), dt.date(2026, 10, 4)):
        assert evidence.weekend_dates(day) == expected
        calendar = evidence.calendar_context(day)
        assert calendar['weekend_dates'] == expected
        assert [calendar['weekdays'][date] for date in expected] == ['Friday', 'Saturday', 'Sunday']


@pytest.mark.parametrize('truncated', [False, True])
def test_large_assessment_bounds_output_and_rejects_truncated_json(monkeypatch, truncated):
    async def generate(_request, body, _user):
        schema = body['response_format']['json_schema']['schema']
        selection = schema['properties']['supported_ids']
        references = selection['items']['enum']
        assert len(references) == 289
        assert selection['maxItems'] == 32
        assert schema['additionalProperties'] is False
        payload = {
            'sufficient': True, 'supported_ids': references[:32],
            'missing': [], 'conflicts': [], 'choices': [], 'clarification_question': None,
        }
        return {'choices': [{
            'finish_reason': 'length' if truncated else 'stop',
            'message': {'content': '{"supported_ids": ["e1",' if truncated else json.dumps(payload)},
        }]}

    module = ModuleType('open_webui.utils.chat')
    module.generate_chat_completion = generate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    research = pipeline.NativeResearch(None, {'model': 'model'}, {'__event_emitter__': None}, None)
    research.question = 'Creative ways to reuse art'
    fitted = [{
        'id': 'original', 'source_id': 'page', 'source': {'name': 'Art ideas'}, 'metadata': {},
        'text': '\n'.join(f'Useful source sentence number {index}.' for index in range(289)),
    }]
    if truncated:
        with pytest.raises(pipeline.ModelOutputTruncated):
            asyncio.run(research.assess(fitted))
    else:
        result = asyncio.run(research.assess(fitted))
        assert result['sufficient'] and result['supported_ids'] == ['original']
        assert len(result['supported_text']['original'].splitlines()) == 32
        assert 'number 32.' not in result['supported_text']['original']


def test_verification_fits_its_actual_payload_before_generating(monkeypatch):
    checked, generated = [], []

    async def generate(_request, body, _user):
        # Stand in for the provider's complete-payload token count. Sentence JSON
        # has more overhead than the same text in the eventual answer prompt.
        serialized = json.dumps(body)
        if len(serialized) > 6500:
            assert context.probing.get(), 'An oversized request reached generation'
            raise ContextBudgetError('local model protected content exceeds context')
        if context.probing.get():
            assert body['messages'][0]['content'] == pipeline.VERIFY
            assert body['max_tokens'] == 1600
            assert body['response_format']['type'] == 'json_schema'
            checked.append(serialized)
            return {'research_context_checked': True, 'payload': body}
        assert serialized in checked, 'The generated request differs from the counted request'
        generated.append(serialized)
        sentences = json.loads(body['messages'][1]['content'])['passages']
        result = {
            'sufficient': True, 'supported_ids': [item['id'] for item in sentences[:32]],
            'missing': [], 'conflicts': [], 'choices': [], 'clarification_question': None,
        }
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}

    module = ModuleType('open_webui.utils.chat')
    module.generate_chat_completion = generate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    request = SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(
        MODELS={'model': {}}, RERANKING_FUNCTION=lambda _query, docs: [0.9] * len(docs),
    )))
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'Explain the requirements'}]}
    research = pipeline.NativeResearch(request, body, {'__event_emitter__': None}, None)
    research.question = research.user_context = body['messages'][0]['content']
    research.sources = [source(
        f'page-{index}',
        ' '.join(f'Page {index} explains installation requirement {number} and its compatibility limits.'
                 for number in range(7)),
        'web',
    ) for index in range(8)]
    asyncio.run(research.evaluate())
    assert research.assessment['sufficient']
    assert 0 < len(research.selected) < len(research.sources)
    assert any(item['reason'] == 'context_limit' for item in research.candidates)
    assert len(generated) == 1
    assert not context.probing.get()


def test_passage_windows_preserve_lines_and_cover_long_input():
    text = '\n'.join(f'[Item {i}](https://example.org/{i})\nItem {i} has its own date and venue.' for i in range(60))
    windows = list(evidence.passage_windows(text))
    assert all(len(window) <= 1000 for _, window in windows)
    for line in text.splitlines():
        assert any(line in window for _, window in windows)
    assert any('[Item 15](https://example.org/15)\nItem 15 has its own date and venue.' in window
               for _, window in windows)


@pytest.mark.parametrize('prior_kind', ['web', 'local'])
@pytest.mark.parametrize('has_addition', [True, False])
def test_expansion_verifies_complementary_source_with_public_prior_answer_only(
    setup, monkeypatch, prior_kind, has_addition
):
    request, emit, events, prompts = setup
    original_model = pipeline.model_json
    original = 'Which backup approaches support restoring a project?'
    reply = 'What other approaches or gaps did that list miss?'
    prior_answer = (
        'A full snapshot can restore the project. Incorrect assistant restriction: snapshots only.'
        if prior_kind == 'web'
        else 'PRIVATE PRIOR: the unreleased project uses a confidential retention policy.'
    )
    repeat = 'A full snapshot can restore the project.'
    addition = 'An incremental backup restores subsequent changes. Retain its matching full baseline.'
    snippet = 'UNVERIFIED SHORTCUT: all backup requirements disappear.'
    previous_answer = prior_answer if prior_kind == 'web' else ''

    async def prepare(_request, body, _user, **_kwargs):
        history = [
            {'role': 'user', 'content': original},
            {'role': 'assistant', 'content': prior_answer},
            {'role': 'user', 'content': reply},
        ]
        previous = {
            'original_query': original,
            'public_answer_context': conversation.public_answer_context(
                history[1], {'report': {'sources': [{'kind': prior_kind, 'selected': True}]}},
            ),
        }
        value = conversation.joint_context(
            {'query': reply, 'original_query': reply, 'source_domains': [], 'cancelled': False},
            history, reply, previous,
        )
        body['metadata']['ravenous_research_context'] = value
        return value

    async def model(*args, **kwargs):
        instruction, data = args[3:5]
        assert 'PRIVATE PRIOR' not in json.dumps(data)
        if instruction == pipeline.PLAN:
            assert data['question'] == 'Which additional backup approaches and recovery gaps were not covered?'
            assert data['conversation'] == [{'role': 'user', 'content': original}]
            assert data['literal_user_context'] == reply
            assert prior_answer not in json.dumps(data)
        if instruction == pipeline.RESOLVE:
            prompts.append((instruction, data))
            return {
                'continuation': True,
                'resolved_intent': 'Which additional backup approaches and recovery gaps were not covered?',
                'conversation_subject': original,
            }
        if instruction == pipeline.VERIFY:
            prompts.append((instruction, data))
            assert data['previous_answer'] == previous_answer
            texts = [item['text'] for item in data['passages']]
            assert repeat in texts
            assert ('An incremental backup restores subsequent changes.' in texts) == has_addition
            assert snippet not in json.dumps(data['passages'])
            return {
                'sufficient': has_addition,
                'supported_ids': [item['id'] for item in data['passages'] if item['text'] != repeat],
                'missing': [] if has_addition else ['No additional verified options.'],
                'conflicts': [], 'choices': [], 'clarification_question': None,
            }
        return await original_model(*args, **kwargs)

    async def batch(_user, _payload, _progress):
        return {
            'calls': 7,
            'pages': [
                {'id': 'p1', 'url': 'https://backup.example/snapshots', 'status': 'read', 'snippet': snippet},
                {'id': 'p2', 'url': 'https://backup.example/incremental', 'status': 'read'},
            ],
            'evidence': [
                {
                    'source': {'url': f'https://backup.example/{slug}', 'title': slug},
                    'text': text, 'fetched_at': '2026-10-03T00:00:00Z', 'content_sha256': digest * 64,
                }
                for slug, text, digest in [('snapshots', repeat, 'a'), ('incremental', addition, 'b')]
                if has_addition or slug == 'snapshots'
            ],
        }

    async def no_local(*_args, **_kwargs):
        return []

    request.app.state.RERANKING_FUNCTION = lambda _query, docs: [
        0.9 if 'full snapshot' in doc.page_content else 0.2 for doc in docs
    ]
    monkeypatch.setattr(pipeline, 'prepare_context', prepare)
    monkeypatch.setattr(pipeline, 'model_json', model)
    monkeypatch.setattr(pipeline.transport, 'batch', batch)
    monkeypatch.setattr(pipeline.local, 'native_sources', no_local)
    monkeypatch.setattr(pipeline.local, 'research_sources', no_local)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': reply}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    generated_context = result['messages'][-1]['content']
    assert snippet not in generated_context and 'PRIVATE PRIOR' not in generated_context
    assert result['metadata']['ravenous_review_previous_answer'] is True
    if has_addition:
        assert json.dumps(previous_answer) in generated_context
        selected = result['metadata']['ravenous_selected_passages']
        assert len(selected) == 1 and selected[0]['source_id'] == 'https://backup.example/incremental'
        assert selected[0]['score'] == 0.2
        assert 'incremental backup' in generated_context and 'matching full baseline' in generated_context
        assert result['metadata']['ravenous_evidence_assessment']['previous_answer'] == previous_answer
        assert not events[-1]['data']['recovery']
    else:
        assert result['metadata']['ravenous_retrieval_sources'] == []
        assert events[-1]['data']['report']['answer_basis'] == 'source_limited'
        assert all(message['role'] != 'system' for message in result['messages'])
        assert generated_context == reply
        assert '<source ' not in generated_context
    assert len([data for instruction, data in prompts if instruction == pipeline.VERIFY]) == 1


def test_rejected_navigation_and_faq_evidence_recovers_without_claiming_coverage(setup, monkeypatch):
    # The model benchmark checks recognition of boilerplate; this checks the
    # pipeline's recovery and answer basis when verification rejects all IDs.
    request, emit, events, _ = setup
    original = pipeline.model_json
    rounds, assessments, recoveries = [], [], []
    boilerplate = 'Home\nExplore archive options\nWhich formats are supported?\nView all possibilities'
    gap = 'No concrete additional archive format is established by these passages.'

    async def model(*args, **kwargs):
        instruction, data = args[3:5]
        if instruction == pipeline.VERIFY:
            assessments.append(data)
            return {
                'sufficient': False,
                'supported_ids': [],
                'missing': [gap],
                'conflicts': [],
                'clarification_question': None,
                'choices': [],
            }
        if instruction not in (pipeline.PLAN, pipeline.RESOLVE):
            recoveries.append(data)
            return {'queries': ['independent archive format capability reference']}
        return await original(*args, **kwargs)

    async def batch(_user, payload, _progress):
        rounds.append(payload['round'])
        return {
            'calls': len(payload['queries']) + 1,
            'pages': [],
            'evidence': [
                {
                    'source': {
                        'url': f'https://example.org/navigation/{payload["round"]}',
                        'title': 'Archive options FAQ',
                    },
                    'text': boilerplate,
                    'fetched_at': '2026-09-19T00:00:00Z',
                    'content_sha256': 'a' * 64,
                }
            ],
        }

    monkeypatch.setattr(pipeline, 'model_json', model)
    monkeypatch.setattr(pipeline.transport, 'batch', batch)
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'What other archive formats are supported?'}]}
    result = asyncio.run(pipeline.run(request, body, {'__event_emitter__': emit}, SimpleNamespace(id='alice')))
    state = result['metadata']['ravenous_research']
    assert rounds == [1, 2] and len(assessments) == 2
    assert len(recoveries) == 1 and recoveries[0]['missing'] == [gap]
    assert not state['sufficient'] and state['answer_basis'] == 'general_knowledge'
    assert state['report']['missing'] == [gap]
    assert state['recovery']['kind'] == 'retry'
    assert result['metadata']['ravenous_retrieval_sources'] == []
    assert state['report']['sources'] and all(
        not item['selected'] and item['citation'] is None for item in state['report']['sources']
    )
    assert boilerplate not in result['messages'][-1]['content']
    assert events[-1]['data']['done'] and events[-1]['data']['recovery']['choices']
