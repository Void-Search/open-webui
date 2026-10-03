"""Unchecked drafts, partial evidence and revoked access cannot become answers."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from open_webui.ravenous_research import answer_review, responses
from starlette.responses import JSONResponse


def passage(kind='web'):
    return {'source_id': 'manual', 'text': 'Journal backups require a matching baseline.',
            'metadata': {'research_kind': kind}}


def test_time_estimates_cannot_borrow_a_number_from_another_source():
    selected = [{'source_id': 'breaks', 'text': 'Take a break for a few minutes.'},
                {'source_id': 'tasks', 'text': 'Try ten tasks, or 10 minutes of data entry.'}]
    assert answer_review.unsupported_numbers('Take a 5–10 minute break. [1]', selected)
    assert answer_review.unsupported_numbers('Take a 10 minute break. [1]', selected)
    assert not answer_review.unsupported_numbers('1. Try 10 minutes of data entry. [2]', selected)
    assert answer_review.unsupported_numbers('The value is 5.5. [1]', [{'source_id': 'value', 'text': 'The value is 5.0.'}])


def test_prior_answer_cannot_supply_a_missing_claim_term_or_version():
    selected = [passage()]
    assert answer_review.borrowed_unsupported_terms(
        'Snapshot backups are supported. [1]', 'Snapshot backups are supported.', selected)
    assert answer_review.borrowed_unsupported_terms(
        'Version 4 supports backups. [1]', 'Version 4 supports backups.', selected)
    assert not answer_review.borrowed_unsupported_terms(
        'Journal backups require the matching baseline. [1]', 'Journal backups use a matching baseline.', selected)
    assert not answer_review.borrowed_unsupported_terms(
        '3. Journal backups require the matching baseline. [1]', '3. What do you need?', selected)


def test_uncertain_inherited_claim_skips_review_and_returns_sources(monkeypatch):
    async def review(*_args):
        raise AssertionError('A copied unsupported claim must fail before model approval')

    async def complete(*_args):
        return {'choices': [{'message': {'content': 'Snapshot backups are supported. [1]'}, 'finish_reason': 'stop'}]}

    monkeypatch.setattr(answer_review, 'check_draft', review)
    selected = [passage()]
    response, quoted = asyncio.run(answer_review.reviewed_response(
        None, {'model': 'fixture'}, None,
        {'ravenous_selected_passages': selected,
         'ravenous_evidence_assessment': {'sufficient': True, 'previous_answer': 'Snapshot backups are supported.'}},
        complete))
    assert quoted and response['choices'][0]['message']['content'] == responses.excerpt_text(selected)


@pytest.mark.parametrize('decision', [True, False, 'true', None])
def test_review_uses_full_passages_and_requires_boolean_approval(monkeypatch, decision):
    selected = [passage()]

    async def fit(_request, _body, _user, items, _question, **kwargs):
        assert kwargs['build_payload'](items)['response_format'] == answer_review.FORMAT
        return items, []

    async def evaluate(_request, _model, _user, _prompt, data, **_kwargs):
        assert data['sources'] == [{'citation': '[1]', 'text': selected[0]['text']}]
        assert 'previous_answer' not in data
        return {'units': [{'id': 'u1', 'evidence': [selected[0]['text']], 'unsupported_details': []}],
                'supported': decision}

    monkeypatch.setattr(answer_review.context, 'fit', fit)
    monkeypatch.setattr(answer_review.pipeline, 'model_json', evaluate)
    assert asyncio.run(answer_review.check_draft(None, {'model': 'fixture'}, None,
                                               'Keep a matching baseline. [1]', selected, 'Explain backups', 20)) is (decision is True)


@pytest.mark.parametrize('review', [
    {'supported': True, 'units': []},
    {'supported': True, 'units': [{'id': 'u1', 'evidence': ['Invented source quote.'], 'unsupported_details': []}]},
    {'supported': True, 'units': [{'id': 'u1', 'evidence': [], 'unsupported_details': []}]},
    {'supported': True, 'units': [{'id': 'u1', 'evidence': ['Journal backups require a matching baseline.'],
                                 'unsupported_details': ['Unsupported activity']}]},
])
def test_missing_or_fabricated_proof_cannot_approve_a_draft(review):
    data = answer_review.review_input('Keep the baseline. [1]', [passage()], 'Explain backups')
    assert answer_review.supported_review(review, data) is False


@pytest.mark.parametrize('draft', ['An uncited claim.', 'An invalid citation. [2]'])
def test_uncited_or_forged_citations_fail_before_review(draft):
    assert asyncio.run(answer_review.check_draft(None, {'model': 'fixture'}, None, draft,
                                               [passage()], 'Explain backups', 20)) is False


@pytest.mark.parametrize('failure', [TimeoutError(), ValueError('Malformed review')])
def test_unavailable_draft_review_falls_back_without_sending_draft(monkeypatch, failure):
    async def check(*_args):
        raise failure

    async def complete(_request, body, _user):
        assert body['stream'] is False
        return {'choices': [{'message': {'content': 'UNSUPPORTED DRAFT [1]'}, 'finish_reason': 'stop'}]}

    monkeypatch.setattr(answer_review, 'check_draft', check)
    body = {'model': 'fixture', 'stream': False}
    selected = [passage()]
    response, quoted = asyncio.run(answer_review.reviewed_response(
        None, body, None, {'ravenous_selected_passages': selected,
                          'ravenous_evidence_assessment': {'sufficient': True},
                          'ravenous_research_context': {'previous_query': 'Earlier research'}}, complete))
    assert quoted and body['stream'] is False
    assert response['choices'][0]['message']['content'] == responses.excerpt_text(selected)
    assert 'UNSUPPORTED DRAFT' not in response['choices'][0]['message']['content']


def test_acl_is_rechecked_after_model_review(monkeypatch):
    async def check(*_args):
        return True

    async def revoked(_user, _selected):
        return {'manual': 'access_revoked'}

    async def complete(*_args):
        return {'choices': [{'message': {'content': 'A private claim. [1]'}, 'finish_reason': 'stop'}]}

    monkeypatch.setattr(answer_review, 'check_draft', check)
    monkeypatch.setattr(answer_review.local, 'authorize_sources', revoked)
    with pytest.raises(HTTPException) as error:
        asyncio.run(answer_review.reviewed_response(
            None, {'model': 'fixture'}, None,
            {'ravenous_selected_passages': [passage('saved')],
             'ravenous_evidence_assessment': {'sufficient': True}}, complete))
    assert error.value.status_code == 403


def test_cancellation_is_not_swallowed():
    async def complete(*_args):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(answer_review.reviewed_response(
            SimpleNamespace(), {'model': 'fixture'}, None,
            {'ravenous_selected_passages': [passage()],
             'ravenous_evidence_assessment': {'sufficient': True}}, complete))


def test_provider_error_keeps_normal_error_handling():
    response = JSONResponse({'error': 'Provider unavailable'}, status_code=503)

    async def complete(*_args):
        return response

    result, quoted = asyncio.run(answer_review.reviewed_response(
        None, {'model': 'fixture'}, None, {'ravenous_selected_passages': [passage()],
                                         'ravenous_evidence_assessment': {'sufficient': True}}, complete))
    assert result is response and quoted is False


@pytest.mark.parametrize('stream', [False, True])
def test_rejected_draft_keeps_targeted_lookup_value_without_neighboring_listing(monkeypatch, stream):
    selected = [{'source_id': 'directory', 'text': '* Willow Hall\n  17 Orchard Road.\n* Cedar Centre\n  22 School Road.',
                 'metadata': {'research_kind': 'web'}}]

    async def check(*_args):
        return False

    async def complete(*_args):
        return {'choices': [{'message': {'content': 'UNSUPPORTED DRAFT [1]'}, 'finish_reason': 'stop'}]}

    monkeypatch.setattr(answer_review, 'check_draft', check)

    async def exercise():
        result, quoted = await answer_review.reviewed_response(
            None, {'model': 'fixture', 'stream': stream}, None,
            {'ravenous_selected_passages': selected,
             'ravenous_research_context': {'previous_query': 'Earlier research',
                                           'referenced_entities': ['Willow Hall', 'Oak Theatre']},
             'ravenous_evidence_assessment': {'sufficient': False, 'verified_details': [
                 {'entity': 'Willow Hall', 'value': '17 Orchard Road.', 'source_id': 'directory'}]}}, complete)
        assert quoted
        if stream:
            chunks = ''.join([chunk async for chunk in result.body_iterator])
            assert chunks.endswith('data: [DONE]\n\n')
            frames = [json.loads(frame.removeprefix('data: ')) for frame in chunks.strip().split('\n\n')[:-1]]
            text = ''.join(frame['choices'][0]['delta'].get('content', '') for frame in frames)
        else:
            text = result['choices'][0]['message']['content']
        assert '17 Orchard Road.' in text and '[1]' in text and 'remaining requested details' in text
        assert 'Cedar' not in text and '22 School' not in text and '> ' not in text and 'UNSUPPORTED' not in text

    asyncio.run(exercise())


def test_compact_lookup_cannot_hide_governing_conditions():
    selected = [{'source_id': 'manual', 'text': 'Only for annual subscriptions.\n\nThe price is $19.'}]
    details = [{'entity': 'Willow', 'value': 'The price is $19.', 'source_id': 'manual'}]
    assert responses.details_text(details, selected) is None
