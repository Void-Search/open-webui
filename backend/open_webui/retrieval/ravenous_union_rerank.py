"""Normalize generated queries and rerank their merged result against user intent."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

log = logging.getLogger(__name__)
_MAX_SENTENCE_GROUPS = 16


def normalize_retrieval_queries(original: str, generated: object, *, limit: int = 3) -> list[str]:
    """Keep the exact original first and discard superficial duplicates."""
    candidates: list[object] = [original]
    if isinstance(generated, list):
        candidates.extend(generated)

    queries = []
    seen = set()
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.strip()
        key = ' '.join(re.findall(r'\w+', candidate.casefold()))
        if key in seen:
            continue
        queries.append(candidate)
        seen.add(key)
        if len(queries) == limit:
            break
    return queries


def prepare_retrieval_queries(
    original: str,
    generated: object,
    standalone_intent: object,
    *,
    limit: int = 3,
) -> tuple[list[str], str]:
    """Return candidate queries and the independently generated scoring intent."""
    queries = normalize_retrieval_queries(original, generated, limit=limit)
    if isinstance(standalone_intent, str) and standalone_intent.strip():
        return queries, standalone_intent.strip()
    return queries, queries[0] if queries else ''


def _enabled() -> bool:
    return os.getenv('RAG_FINAL_UNION_RERANKING', 'false').lower() == 'true'


def _scoring_windows(document: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', document) if part.strip()]
    if len(sentences) <= 1:
        return [document]
    group_size = max(1, (len(sentences) + _MAX_SENTENCE_GROUPS - 1) // _MAX_SENTENCE_GROUPS)
    groups = [' '.join(sentences[index : index + group_size]) for index in range(0, len(sentences), group_size)]
    return groups + [f'{first} {second}' for first, second in pairwise(groups)]


async def _score_documents(
    query: str,
    documents: list[str],
    metadatas: list[dict[str, Any]],
    reranking_function: Callable[[str, list[Any]], Any],
) -> list[tuple[float, str]]:
    scoring_documents = []
    document_indexes = []
    for document_index, (document, metadata) in enumerate(zip(documents, metadatas)):
        for window in _scoring_windows(document):
            scoring_documents.append(SimpleNamespace(page_content=window, metadata=metadata))
            document_indexes.append(document_index)
    raw_scores = await asyncio.to_thread(reranking_function, query, scoring_documents)
    scores = raw_scores.tolist() if hasattr(raw_scores, 'tolist') else raw_scores
    if not isinstance(scores, list) or len(scores) != len(scoring_documents):
        raise ValueError('reranker returned an unexpected score count')
    best_windows = [(float('-inf'), document) for document in documents]
    for score, document_index, scoring_document in zip(scores, document_indexes, scoring_documents):
        score = float(score)
        if score > best_windows[document_index][0]:
            best_windows[document_index] = (score, scoring_document.page_content)
    return best_windows


async def rerank_merged_result(
    result: dict[str, Any],
    *,
    query: str,
    reranking_function: Callable[[str, list[Any]], Any] | None,
    limit: int,
    relevance_threshold: float,
) -> dict[str, Any]:
    """Rerank a deduplicated result against intent and return focused evidence windows."""
    if not _enabled() or not query or reranking_function is None:
        return result

    documents = (result.get('documents') or [[]])[0]
    metadatas = (result.get('metadatas') or [[]])[0]
    if not documents or len(documents) != len(metadatas):
        return result

    try:
        scored_windows = await _score_documents(query, documents, metadatas, reranking_function)
        ranked = sorted(
            (
                (score, window, metadata)
                for (score, window), metadata in zip(scored_windows, metadatas)
                if not relevance_threshold or float(score) >= relevance_threshold
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:limit]
    except Exception:  # pylint: disable=broad-exception-caught
        log.exception('Final union reranking failed; retaining merged query results')
        return result

    return {
        **result,
        'distances': [[item[0] for item in ranked]],
        'documents': [[item[1] for item in ranked]],
        'metadatas': [[item[2] for item in ranked]],
    }
