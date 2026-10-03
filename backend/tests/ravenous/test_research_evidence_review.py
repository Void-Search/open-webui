"""Score-threshold review preserves same-source details without bypassing exclusions."""

import asyncio

import pytest
from open_webui.ravenous_research import evidence


def test_strong_summary_does_not_hide_additional_details_on_the_same_page():
    texts = [
        'The backup guide summarizes snapshot export.',
        'A journal can replay committed changes since a matching baseline.',
        'Keep the baseline and journal together to restore the complete state.',
    ]
    candidates = evidence.passages([{
        'source': {'id': 'guide', 'name': 'Recovery guide'},
        'document': texts,
        'metadata': [{'source': 'guide', 'research_kind': 'web'} for _ in texts],
    }])
    ordered, failed = asyncio.run(evidence.rank(
        'What additional recovery methods and limitations were missed?',
        candidates, lambda _query, _docs: [0.9, 0.3, 0.2],
    ))
    assert not failed and len(ordered) == 1
    reviewed = evidence.supplement_sources(ordered, candidates)
    assert [item['text'] for item in reviewed] == texts
    assert [item['score'] for item in reviewed] == [0.9, 0.3, 0.2]
    assert all(item['reason'] is None for item in reviewed)


@pytest.mark.parametrize('limit', [0, 1, 3, 12])
def test_low_score_review_has_one_global_limit_and_cycles_across_sources(limit):
    candidates = [
        {'id': f'{source}-{index}', 'source_id': source, 'score': score, 'reason': 'low_relevance'}
        for source, score in [('long-page', 0.3), ('short-page', 0.1)]
        for index in range(20 if source == 'long-page' else 2)
    ]
    excluded = [
        {'id': reason, 'source_id': 'excluded', 'score': 0.39, 'reason': reason}
        for reason in ('duplicate', 'source_outside_scope', 'candidate_limit', 'access_revoked')
    ]
    reviewed = evidence.review_low_scores([*candidates, *excluded], limit=limit)
    assert len(reviewed) == limit
    if limit >= 3:
        assert [item['id'] for item in reviewed[:3]] == ['long-page-0', 'short-page-0', 'long-page-1']
    assert not any(item in reviewed for item in excluded)
    assert all(item['reason'] == item['id'] for item in excluded)
    assert sum(item['reason'] is None for item in candidates) == limit
    assert all(item['reason'] == 'low_relevance' for item in candidates if item not in reviewed)


def test_review_additions_precede_repeated_strong_windows_without_displacing_source_leads():
    strong = [
        {'id': identity, 'source_id': source, 'score': 0.9, 'reason': None}
        for identity, source in [('lead-a', 'a'), ('lead-b', 'b'), ('more-a', 'a'), ('more-b', 'b')]
    ]
    weak = [
        {'id': identity, 'source_id': source, 'score': score, 'reason': 'low_relevance'}
        for identity, source, score in [('detail-a', 'a', 0.3), ('lead-c', 'c', 0.2)]
    ]
    reviewed = evidence.supplement_sources(strong, [*strong, *weak], limit=2)
    assert [item['id'] for item in reviewed] == [
        'lead-a', 'lead-b', 'detail-a', 'lead-c', 'more-a', 'more-b',
    ]
    assert len({item['id'] for item in reviewed}) == len(reviewed)
