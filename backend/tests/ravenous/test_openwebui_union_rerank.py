"""Tests for final multi-query union reranking."""

import asyncio

from open_webui.retrieval.ravenous_union_rerank import (
    normalize_retrieval_queries,
    prepare_retrieval_queries,
    rerank_merged_result,
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

    assert result == merged
