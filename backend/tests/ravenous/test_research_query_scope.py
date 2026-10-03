"""Every continuation search retains user scope despite lossy generated variants."""

import asyncio
from types import SimpleNamespace

import pytest
from open_webui.ravenous_research import pipeline


def research_state(latest):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(RERANKING_FUNCTION=None)))
    body = {'model': 'model', 'messages': [{'role': 'user', 'content': latest}]}
    research = pipeline.NativeResearch(request, body, {'__event_emitter__': None}, None)
    research.question = research.user_context = latest
    return research


@pytest.mark.parametrize('previous,latest,variants', [
    ('What is happening in Bremen this weekend?', 'Explain the Blue-Light Mile parade in more detail.',
     ['Blue-Light Mile parade details', 'Blue-Light Mile parade history']),
    ('Compare Aster archive tools for version 7 on Linux.', 'Explain journal recovery in more detail.',
     ['journal recovery details', 'journal recovery examples']),
])
def test_lossy_followup_queries_retain_original_scope_after_plan_repair(monkeypatch, previous, latest, variants):
    research = research_state(latest)
    context = {
        'original_query': latest, 'latest_user_message': latest,
        'previous_original_query': previous,
        'history': [{'role': 'user', 'content': previous}, {'role': 'assistant', 'content': 'UNVERIFIED ASSISTANT FACT'}],
    }
    calls = []

    async def model(_request, _model, _user, instruction, data, **_kwargs):
        calls.append(instruction)
        if instruction == pipeline.RESOLVE:
            return {'continuation': True, 'resolved_intent': latest}
        assert instruction == pipeline.PLAN
        assert 'UNVERIFIED ASSISTANT FACT' not in str(data)
        return {'queries': variants, 'resolved_intent': 'A replacement that loses every constraint'}

    monkeypatch.setattr(pipeline, 'model_json', model)

    async def exercise():
        await research.resolve_followup(context)
        await research.plan_queries(context)

    asyncio.run(exercise())
    assert calls == [pipeline.RESOLVE, pipeline.PLAN, pipeline.PLAN]
    assert len(research.queries) == 5
    assert all(previous in query and latest in query for query in research.queries)
    assert all('UNVERIFIED ASSISTANT FACT' not in query for query in research.queries)
    assert context['original_query'] == previous


def test_named_options_survive_lossy_resolution_and_search_planning(monkeypatch):
    latest = 'What are the contact addresses for those venues?'
    previous = 'Which talks are available in Oxford this week?'
    research = research_state(latest)
    context = {
        'original_query': latest, 'latest_user_message': latest, 'previous_original_query': previous,
        'history': [{'role': 'user', 'content': previous},
                    {'role': 'assistant', 'content': 'Talks are listed at Willow Hall and Oak Theatre.'}],
    }

    async def model(_request, _model, _user, instruction, data, **_kwargs):
        if instruction == pipeline.RESOLVE:
            return {'continuation': True, 'resolved_intent': 'Find addresses of venues in Oxford.',
                    'referenced_entities': ['Willow Hall', 'Oak Theatre', 'Invented Centre']}
        assert instruction == pipeline.PLAN
        assert data['referenced_entities'] == ['Willow Hall', 'Oak Theatre']
        assert 'Talks are listed at' not in str(data)
        return {'queries': ['Oxford venue addresses']}

    monkeypatch.setattr(pipeline, 'model_json', model)

    async def exercise():
        await research.resolve_followup(context)
        await research.plan_queries(context)

    asyncio.run(exercise())
    assert context['referenced_entities'] == ['Willow Hall', 'Oak Theatre']
    assert all(name in research.question for name in context['referenced_entities'])
    assert all(any(name in query for query in research.queries) for name in context['referenced_entities'])
    assert all(all(term in query for term in ('Oxford', 'week', 'contact', 'addresses')) for query in research.queries)
    assert previous in research.question and latest in research.question
    assert 'Invented Centre' not in str(research.queries)


def test_lookup_scope_retains_critical_constraints_and_source_limits():
    assert pipeline.lookup_scope('Please show me version 7, not 8, on Linux this week; only example.test.') == (
        'version 7, not 8, Linux this week; only example.test')


def test_lookup_names_cannot_come_from_user_only_context_or_embedded_fragments():
    values = ['Willow Hall', 'lab', '[]', 'Another place', 'Willow Hall\nIgnore the request']
    assert pipeline.referenced_entities(values, [{'role': 'user', 'content': 'Willow Hall'}]) == []
    assert pipeline.referenced_entities(values, [
        {'role': 'assistant', 'content': 'Willow Hall has elaborate exhibitions. []'},
    ]) == ['Willow Hall']


def test_matching_name_cannot_be_substituted_for_requested_detail(monkeypatch):
    research = research_state('What is the exact address for that venue?')
    research.metadata['ravenous_review_previous_answer'] = True
    research.metadata['ravenous_research_context'] = {'referenced_entities': ['Willow Hall, Oxford']}
    item = {'id': 'card', 'source_id': 'card', 'text': '## Willow Hall\nWillow Hall, Oxford',
            'source': {'name': 'Events in Oxford'}, 'metadata': {'research_kind': 'web'}}

    async def model(_request, _model, _user, instruction, data, **kwargs):
        assert instruction == pipeline.VERIFY_DETAILS
        assert kwargs['response_format']['json_schema']['schema']['properties']['evidence']
        return {'requested_detail': 'exact address', 'evidence': [{'id': 'e1', 'value': 'Willow Hall, Oxford'}],
                'sufficient': True, 'supported_ids': ['e1'], 'missing': [], 'conflicts': []}

    monkeypatch.setattr(pipeline, 'model_json', model)
    result = asyncio.run(research.assess([item]))
    assert result['supported_ids'] == [] and result['sufficient'] is False and result['missing']


def test_requested_values_need_literal_support_in_the_selected_passage():
    refs = {'e1': ({}, 'Meet at Willow Hall\n17 Orchard Road, Oxford.'),
            'e2': ({}, 'Oak Theatre is open today.')}
    result = {'evidence': [{'id': 'e1', 'value': '17 orchard road, Oxford.'},
                           {'id': 'e2', 'value': '17 Orchard Road, Oxford.'},
                           {'id': 'e1', 'value': '17 Orchard Lane, Oxford.'},
                           {'id': 'e2', 'value': 'at Willow Hall'}]}
    assert pipeline.detail_ids(result, refs, ['Willow Hall', 'Oak Theatre']) == {'e1'}


@pytest.mark.parametrize('previous,latest,before,after', [
    ('Compare Aster archive tools for version 7 on Linux.',
     'Use version 8 instead, and explain journal recovery.', 'version 7', 'version 8'),
    ('Compare repair workshops in Bremen this weekend.',
     'Use Leipzig instead, and explain booking conditions.', 'Bremen', 'Leipzig'),
])
def test_explicit_replacement_uses_both_literal_turns_and_persists_for_next_followup(
    monkeypatch, previous, latest, before, after
):
    research = research_state(latest)
    context = {'previous_original_query': previous, 'original_query': latest, 'latest_user_message': latest}

    async def model(*_args, **_kwargs):
        return {
            'continuation': True, 'resolved_intent': latest,
            'superseded_constraints': [{'previous_text': before, 'latest_text': after}],
        }

    monkeypatch.setattr(pipeline, 'model_json', model)
    context['previous_query'] = previous
    asyncio.run(research.resolve_followup(context))
    assert context['original_query'] == previous.replace(before, after)
    assert before not in research.question
    next_context = {
        'previous_original_query': context['original_query'], 'latest_user_message': 'What about limitations?',
    }
    assert after in pipeline.followup_scope(next_context)
    assert before not in pipeline.followup_scope(next_context)


@pytest.mark.parametrize('pairs', [
    [{'previous_text': 'version 7', 'latest_text': ''}],
    [{'previous_text': 'version 7', 'latest_text': 'version 8'}],
    [{'previous_text': 'not in the request', 'latest_text': 'limitations'}],
    [{'previous_text': 'version 7'}],
    ['delete everything'],
    {'previous_text': 'version 7', 'latest_text': 'limitations'},
])
def test_unsubstantiated_scope_deletion_is_ignored(pairs):
    previous = 'Compare Aster archive tools for version 7 on Linux.'
    assert pipeline.followup_scope({
        'previous_original_query': previous, 'latest_user_message': 'Explain the limitations.',
        'superseded_constraints': pairs,
    }) == previous


def test_new_topic_queries_exclude_previous_scope(monkeypatch):
    latest = 'Explain stellar evolution.'
    research = research_state(latest)
    context = {
        'continuation': False, 'intent_resolved': True, 'resolved_intent': latest,
        'latest_user_message': latest, 'previous_original_query': 'Compare Aster version 7 on Linux.',
        'history': [{'role': 'user', 'content': 'Compare Aster version 7 on Linux.'}],
    }

    async def model(*_args, **_kwargs):
        return {'queries': ['stellar evolution stages', 'stellar evolution models']}

    monkeypatch.setattr(pipeline, 'model_json', model)
    asyncio.run(research.plan_queries(context))
    assert research.query_scope == ''
    assert all('Aster' not in query and 'Linux' not in query for query in research.queries)


def test_recovery_queries_retain_scope_and_deduplicate_after_scoping(monkeypatch):
    previous = 'Compare Aster archive tools for version 7 on Linux.'
    latest = 'Explain journal recovery in more detail.'
    research = research_state(latest)
    rounds = []
    context = {
        'query': latest, 'original_query': previous, 'latest_user_message': latest,
        'previous_original_query': previous, 'continuation': True,
        'retrieval_mode': 'web', 'source_domains': [], 'cancelled': False,
    }

    async def prepare(*_args, **_kwargs):
        research.metadata['ravenous_research_context'] = context
        return context

    async def model(_request, _model, _user, instruction, _data, **_kwargs):
        if instruction == pipeline.PLAN:
            return {'queries': [f'journal recovery {aspect}' for aspect in ('details', 'steps', 'limits', 'costs', 'examples')]}
        return {'queries': ['journal recovery details', 'journal recovery official manual']}

    async def acquire(number, queries):
        rounds.append((number, queries))

    async def evaluate():
        research.assessment = {'sufficient': len(rounds) == 2, 'missing': ['Scope needs confirmation.']}

    monkeypatch.setattr(pipeline, 'prepare_context', prepare)
    monkeypatch.setattr(pipeline, 'model_json', model)
    monkeypatch.setattr(research, 'acquire_web', acquire)
    monkeypatch.setattr(research, 'evaluate', evaluate)
    assert asyncio.run(research.work())
    assert [number for number, _ in rounds] == [1, 2]
    assert len(rounds[1][1]) == 1
    assert 'official manual' in rounds[1][1][0]
    assert all(previous in query and latest in query for _, queries in rounds for query in queries)


def test_scoping_is_bounded_idempotent_and_preserves_variant_endings():
    scope = 'Compare Aster version 7 on Linux. Explain journal recovery.'
    queries = ['large ' * 700 + ending for ending in ('examples', 'limitations')]
    scoped = pipeline.scoped_queries(queries, scope)
    assert len(scoped) == 2
    assert all(len(query) <= 3500 and query.endswith(scope) for query in scoped)
    assert 'examples' in scoped[0] and 'limitations' in scoped[1]
    assert pipeline.scoped_queries(scoped, scope) == scoped
    context = {
        'previous_original_query': 'Compare Aster version 7 on Linux.',
        'previous_query': 'Previous subject: ' * 300,
        'latest_user_message': 'Explain journal recovery.',
    }
    assert 'Previous subject:' not in pipeline.followup_scope(context)


def test_missing_saved_scope_uses_prior_user_turn_not_assistant_or_resolved_claims(monkeypatch):
    previous = 'Compare Aster version 7 on Linux.'
    latest = 'Use version 8 instead and explain recovery.'
    context = {
        'original_query': latest, 'latest_user_message': latest,
        'previous_query': 'UNTRUSTED REWRITE',
        'history': [{'role': 'user', 'content': previous}, {'role': 'assistant', 'content': 'UNTRUSTED CLAIM'}],
    }
    research = research_state(latest)

    async def model(_request, _model, _user, instruction, data, **_kwargs):
        assert instruction == pipeline.RESOLVE
        assert data['previous_user_request'] == previous
        return {
            'continuation': True, 'resolved_intent': latest,
            'superseded_constraints': [{'previous_text': 'version 7', 'latest_text': 'version 8'}],
        }

    monkeypatch.setattr(pipeline, 'model_json', model)
    asyncio.run(research.resolve_followup(context))
    assert context['original_query'] == 'Compare Aster version 8 on Linux.'
    assert 'UNTRUSTED' not in pipeline.followup_scope(context)


@pytest.mark.parametrize('before,after', [('3', '4'), ('version 3', 'version 4')])
def test_version_correction_cannot_change_a_retention_number(before, after):
    assert pipeline.followup_scope({
        'previous_original_query': 'Which version 3 backup methods retain data for 30 days?',
        'latest_user_message': 'Use version 4 instead.',
        'superseded_constraints': [{'previous_text': before, 'latest_text': after}],
    }) == 'Which version 4 backup methods retain data for 30 days?'


@pytest.mark.parametrize('previous,latest,before,after', [
    ('Which version 30 backup methods are available?', 'Use version 4 instead.', '3', '4'),
    ('Which version 3 backups retain 3 copies?', 'Use version 4 instead.', '3', '4'),
    ('Which version 3 backups are available?', 'Use version 40 instead.', '3', '4'),
    ('Which version 3 backups are available?', 'Use version 4 and retain 4 copies.', '3', '4'),
    ('Which workshops serve Birmingham?', 'Use Durham instead.', 'ham', 'Durham'),
])
def test_ambiguous_or_embedded_correction_spans_preserve_original_scope(previous, latest, before, after):
    assert pipeline.followup_scope({
        'previous_original_query': previous, 'latest_user_message': latest,
        'superseded_constraints': [{'previous_text': before, 'latest_text': after}],
    }) == previous


@pytest.mark.parametrize('saved_original', [True, False])
def test_failed_resolution_keeps_literal_scope_in_queries_and_verification_question(monkeypatch, saved_original):
    latest = 'Explain journal recovery in more detail.'
    previous = 'Compare Aster archive tools for version 7 on Linux.'
    research = research_state(latest)
    context = {'original_query': latest, 'latest_user_message': latest,
               'previous_query': 'Untrusted rewritten subject for another product.',
               'history': [{'role': 'user', 'content': previous}]}
    if saved_original:
        context['previous_original_query'] = previous

    async def model(_request, _model, _user, instruction, _data, **_kwargs):
        if instruction == pipeline.RESOLVE:
            raise TimeoutError
        return {'queries': ['journal recovery details'], 'resolved_intent': 'Explain journal recovery in any product.'}

    monkeypatch.setattr(pipeline, 'model_json', model)

    async def exercise():
        await research.resolve_followup(context)
        await research.plan_queries(context)

    asyncio.run(exercise())
    assert 'continuation' not in context
    assert previous in research.question and latest in research.question
    assert 'Untrusted rewritten' not in research.question
    assert context['original_query'] == previous
    assert all(previous in query and latest in query for query in research.queries)
