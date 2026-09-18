"""Tests for final multi-query union reranking."""

import asyncio
from typing import Any

import pytest
from open_webui.retrieval.ravenous_union_rerank import (
    RETRIEVAL_ASSESSMENT_KEY,
    filter_retrieval_sources,
    load_clarification_threshold,
    normalize_retrieval_queries,
    prepare_retrieval_queries,
    rerank_merged_result,
    retrieval_response_instruction,
    summarize_retrieval_sources,
)


def test_normalize_retrieval_queries_preserves_exact_original():
    queries = normalize_retrieval_queries(
        'Original CASTING question?',
        ['original casting question?', 'compact casting relation', 'source wording'],
    )

    assert queries == [
        'Original CASTING question?',
        'compact casting relation',
        'source wording',
    ]


def test_prepare_retrieval_queries_separates_followup_from_standalone_intent():
    queries, rerank_query = prepare_retrieval_queries(
        'what about just for the lord of the rings?',
        [
            'what about just for the lord of the rings',
            'alternative casting choices for the wizard in the film trilogy',
            'which performers were considered for the same role in those films',
        ],
        'Which performers were alternative casting choices for the wizard in the film trilogy?',
    )

    assert queries == [
        'what about just for the lord of the rings?',
        'alternative casting choices for the wizard in the film trilogy',
        'which performers were considered for the same role in those films',
    ]
    assert rerank_query == ('Which performers were alternative casting choices for the wizard in the film trilogy?')


def test_prepare_retrieval_queries_falls_back_to_exact_original():
    queries, rerank_query = prepare_retrieval_queries('exact request', ['expanded request'], None)

    assert queries == ['exact request', 'expanded request']
    assert rerank_query == 'exact request'


def test_rerank_merged_result_uses_original_intent(monkeypatch):
    monkeypatch.setenv('RAG_FINAL_UNION_RERANKING', 'true')
    merged = {
        'documents': [['different portrayals', 'actors declined the offered role']],
        'metadatas': [[{'id': 'portrayals'}, {'id': 'casting'}]],
        'distances': [[0.99, 0.80]],
    }

    def score_documents(query, documents):
        assert query == 'Who were the alternative casting choices?'
        assert [document.page_content for document in documents] == [
            'different portrayals',
            'actors declined the offered role',
        ]
        return [0.2, 0.9]

    result = asyncio.run(
        rerank_merged_result(
            merged,
            query='Who were the alternative casting choices?',
            reranking_function=score_documents,
            limit=1,
            relevance_threshold=0.0,
        )
    )

    assert result['documents'] == [['actors declined the offered role']]
    assert result['metadatas'] == [[{'id': 'casting'}]]
    assert result['distances'] == [[0.9]]


def test_rerank_merged_result_scores_sentence_windows(monkeypatch):
    monkeypatch.setenv('RAG_FINAL_UNION_RERANKING', 'true')
    merged = {
        'documents': [
            [
                'The subject appeared elsewhere. A performer portrayed the role.',
                'General background. Two candidates declined the offered role.',
            ]
        ],
        'metadatas': [[{'id': 'portrayals'}, {'id': 'selection'}]],
    }

    def score_windows(_query, documents):
        return [0.95 if 'declined the offered role' in document.page_content else 0.2 for document in documents]

    result = asyncio.run(
        rerank_merged_result(
            merged,
            query='Which candidates were considered for the role?',
            reranking_function=score_windows,
            limit=1,
            relevance_threshold=0.0,
        )
    )

    assert result['documents'] == [['Two candidates declined the offered role.']]
    assert result['distances'] == [[0.95]]


def test_rerank_merged_result_retains_union_when_scorer_fails(monkeypatch):
    monkeypatch.setenv('RAG_FINAL_UNION_RERANKING', 'true')
    merged = {
        'documents': [['first', 'second']],
        'metadatas': [[{'id': 'first'}, {'id': 'second'}]],
        'distances': [[0.9, 0.8]],
    }

    def fail_reranking(_query, _documents):
        raise OSError('scorer unavailable')

    result = asyncio.run(
        rerank_merged_result(
            merged,
            query='original intent',
            reranking_function=fail_reranking,
            limit=1,
            relevance_threshold=0.0,
        )
    )

    assert result['documents'] == merged['documents']
    assert result[RETRIEVAL_ASSESSMENT_KEY]['status'] == 'failed'


def test_rerank_rejects_unnormalized_scores(monkeypatch):
    monkeypatch.setenv('RAG_FINAL_UNION_RERANKING', 'true')
    merged = {
        'documents': [['candidate']],
        'metadatas': [[{'id': 'candidate'}]],
        'distances': [[0.8]],
    }

    result = asyncio.run(
        rerank_merged_result(
            merged,
            query='intent',
            reranking_function=lambda _query, _documents: [1.2],
            limit=1,
            relevance_threshold=0.0,
        )
    )

    assert result[RETRIEVAL_ASSESSMENT_KEY]['status'] == 'failed'
    assert result[RETRIEVAL_ASSESSMENT_KEY]['has_documents'] is True


def test_filtered_or_empty_results_request_clarification(monkeypatch):
    monkeypatch.setenv('RAG_FINAL_UNION_RERANKING', 'true')
    merged = {
        'documents': [['weak candidate']],
        'metadatas': [[{'id': 'weak'}]],
        'distances': [[0.8]],
    }
    filtered = asyncio.run(
        rerank_merged_result(
            merged,
            query='intent',
            reranking_function=lambda _query, _documents: [0.4],
            limit=1,
            relevance_threshold=0.5,
        )
    )
    source = {RETRIEVAL_ASSESSMENT_KEY: filtered[RETRIEVAL_ASSESSMENT_KEY]}

    assert filtered['documents'] == [[]]
    assert summarize_retrieval_sources([source], 0.5) == {
        'status': 'empty',
        'clarify': True,
    }


def test_best_final_score_controls_strict_cutoff():
    def source(score):
        return {RETRIEVAL_ASSESSMENT_KEY: {'status': 'scored', 'best_score': score}}

    assert summarize_retrieval_sources([source(0.49)], 0.5)['clarify'] is True
    assert summarize_retrieval_sources([source(0.5)], 0.5)['clarify'] is False
    mixed = summarize_retrieval_sources([source(0.2), source(0.8)], 0.5)
    assert mixed == {'status': 'scored', 'best_score': 0.8, 'clarify': False}


def test_unscored_sources_do_not_trigger_clarification():
    source = {RETRIEVAL_ASSESSMENT_KEY: {'status': 'unscored'}}

    assessment = summarize_retrieval_sources([source], 0.5)

    assert assessment == {'status': 'unscored', 'clarify': False}
    assert retrieval_response_instruction(assessment) is None


def test_failure_uses_availability_instruction_instead_of_clarification():
    source = {
        RETRIEVAL_ASSESSMENT_KEY: {
            'status': 'failed',
            'reason': 'reranker_failed',
        }
    }

    assessment = summarize_retrieval_sources([source], 0.5)
    instruction = retrieval_response_instruction(assessment)

    assert assessment == {'status': 'failed', 'clarify': False, 'partial': False}
    assert 'could not be retrieved or reranked' in instruction
    assert 'no answer exists' in instruction
    assert 'one to three' not in instruction


def test_weak_evidence_instruction_asks_targeted_questions():
    assessment = {'status': 'scored', 'best_score': 0.4, 'clarify': True}

    instruction = retrieval_response_instruction(assessment)

    assert 'one to three concise questions' in instruction
    assert 'entity, version, date, scope, or document' in instruction
    assert 'model knowledge' in instruction


def test_empty_results_still_produce_clarification_instruction():
    instruction = retrieval_response_instruction({'status': 'empty', 'clarify': True})

    assert 'one to three concise questions' in instruction
    assert 'did not yield sufficiently relevant evidence' in instruction


def test_clarification_threshold_defaults_and_validates(monkeypatch):
    monkeypatch.delenv('RAVENOUS_RAG_CLARIFICATION_THRESHOLD', raising=False)
    assert load_clarification_threshold() == 0.4

    monkeypatch.setenv('RAVENOUS_RAG_CLARIFICATION_THRESHOLD', '0.75')
    assert load_clarification_threshold() == 0.75

    monkeypatch.setenv('RAVENOUS_RAG_CLARIFICATION_THRESHOLD', '1.1')
    with pytest.raises(ValueError, match='between 0 and 1'):
        load_clarification_threshold()


@pytest.mark.parametrize("status", ["empty", "scored"])
@pytest.mark.parametrize("explicit_unscored", [False, True])
def test_independent_documents_survive_weak_retrieval(status, explicit_unscored):
    attachment: dict[str, Any] = {"document": ["The attached report answers the question."]}
    if explicit_unscored:
        attachment[RETRIEVAL_ASSESSMENT_KEY] = {"status": "unscored"}
    weak = {
        "document": ["Unrelated passage"] if status == "scored" else [],
        RETRIEVAL_ASSESSMENT_KEY: {"status": status, "best_score": 0.1},
    }
    sources = [attachment, weak]
    assessment = summarize_retrieval_sources(sources, 0.4)
    assert assessment is not None and assessment["clarify"] is False
    assert retrieval_response_instruction(assessment) is None
    assert filter_retrieval_sources(sources, 0.4) == [attachment]


def test_weak_retrieval_alone_still_clarifies_after_filtering():
    sources = [
        {
            "document": ["Weak passage"],
            RETRIEVAL_ASSESSMENT_KEY: {
                "status": "scored",
                "best_score": 0.1,
            },
        }
    ]
    assessment = summarize_retrieval_sources(sources, 0.4)
    assert not filter_retrieval_sources(sources, 0.4)
    assert assessment is not None and assessment["clarify"] is True
