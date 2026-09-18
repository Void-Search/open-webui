"""Normalize generated queries and rerank their merged result against user intent."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections.abc import Callable
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

log = logging.getLogger(__name__)
_MAX_SENTENCE_GROUPS = 16
RETRIEVAL_ASSESSMENT_KEY = "_ravenous_retrieval_assessment"
_DEFAULT_CLARIFICATION_THRESHOLD = 0.4


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
        key = " ".join(re.findall(r"\w+", candidate.casefold()))
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
    return queries, queries[0] if queries else ""


def _enabled() -> bool:
    return os.getenv("RAG_FINAL_UNION_RERANKING", "false").lower() == "true"


def load_clarification_threshold() -> float:
    """Load the validated host threshold used only for final reranker scores."""
    value = float(
        os.getenv(
            "RAVENOUS_RAG_CLARIFICATION_THRESHOLD",
            str(_DEFAULT_CLARIFICATION_THRESHOLD),
        )
    )
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("clarification threshold must be between 0 and 1")
    return value


def _assessed_result(
    result: dict[str, Any],
    status: str,
    *,
    best_score: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    assessment: dict[str, Any] = {"status": status}
    if best_score is not None:
        assessment["best_score"] = best_score
    if reason is not None:
        assessment["reason"] = reason
    documents = (result.get("documents") or [[]])[0]
    if status == "failed" and documents:
        assessment["has_documents"] = True
    return {**result, RETRIEVAL_ASSESSMENT_KEY: assessment}


def failed_retrieval_result(result: dict[str, Any], reason: str) -> dict[str, Any]:
    """Mark a local retrieval result as unavailable without exposing an exception."""
    return _assessed_result(result, "failed", reason=reason)


def summarize_retrieval_sources(
    sources: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    """Summarize internal final-rerank assessments for the answer stage."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("clarification threshold must be between 0 and 1")
    assessments: list[dict[str, Any]] = []
    independent_documents = False
    for source in sources:
        assessment = source.get(RETRIEVAL_ASSESSMENT_KEY)
        if isinstance(assessment, dict):
            assessments.append(assessment)
        if (not isinstance(assessment, dict) or assessment.get("status") == "unscored") and any(
            source.get("document") or []
        ):
            independent_documents = True
    if not assessments:
        return None

    failures = [item for item in assessments if item.get("status") == "failed"]
    scores = [
        float(item["best_score"])
        for item in assessments
        if item.get("status") == "scored"
        and isinstance(item.get("best_score"), (float, int))
        and not isinstance(item.get("best_score"), bool)
        and math.isfinite(float(item["best_score"]))
        and 0 <= float(item["best_score"]) <= 1
    ]
    if failures:
        return {
            "status": "failed",
            "clarify": False,
            "partial": independent_documents
            or bool(scores)
            or any(item.get("has_documents") for item in failures),
        }
    if scores:
        best_score = max(scores)
        return {
            "status": "scored",
            "best_score": best_score,
            "clarify": best_score < threshold and not independent_documents,
        }
    if any(item.get("status") == "empty" for item in assessments):
        return {"status": "empty", "clarify": not independent_documents}
    return {"status": "unscored", "clarify": False}


def filter_retrieval_sources(
    sources: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Drop weak assessed passages while preserving independently supplied evidence."""
    retained = []
    for source in sources:
        assessment = summarize_retrieval_sources([source], threshold)
        if not assessment or not assessment.get("clarify"):
            retained.append(source)
    return retained


def retrieval_response_instruction(assessment: dict[str, Any] | None) -> str | None:
    """Return answer-stage guidance without adding another model invocation."""
    if not assessment:
        return None
    if assessment.get("status") == "failed":
        scope = "Some selected sources" if assessment.get("partial") else "Selected knowledge"
        return (
            "<retrieval_status>"
            f"{scope} could not be retrieved or reranked. Do not describe this as proof that "
            "no answer exists. Briefly explain the availability problem. If other supplied "
            "sources directly support the answer, use only that evidence; otherwise ask the "
            "user to retry."
            "</retrieval_status>"
        )
    if assessment.get("clarify"):
        return (
            "<retrieval_status>"
            "The selected knowledge did not yield sufficiently relevant evidence. Do not answer "
            "the factual request from model knowledge or guess. Say what evidence is missing, then "
            "ask one to three concise questions about the needed entity, version, date, scope, or "
            "document. Offer choices only when the conversation establishes them, and do not imply "
            "that an already specific request was poorly worded."
            "</retrieval_status>"
        )
    return None


def _scoring_windows(document: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", document) if part.strip()]
    if len(sentences) <= 1:
        return [document]
    group_size = max(1, (len(sentences) + _MAX_SENTENCE_GROUPS - 1) // _MAX_SENTENCE_GROUPS)
    groups = [
        " ".join(sentences[index : index + group_size])
        for index in range(0, len(sentences), group_size)
    ]
    return groups + [f"{first} {second}" for first, second in pairwise(groups)]


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
    scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if not isinstance(scores, list) or len(scores) != len(scoring_documents):
        raise ValueError("reranker returned an unexpected score count")
    best_windows = [(float("-inf"), document) for document in documents]
    for score, document_index, scoring_document in zip(scores, document_indexes, scoring_documents):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("reranker returned a score outside the normalized range")
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
        return _assessed_result(result, "unscored")

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    if not documents:
        return _assessed_result(result, "empty")
    if len(documents) != len(metadatas):
        return _assessed_result(result, "failed", reason="invalid_result_shape")

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
        log.exception("Final union reranking failed; retaining merged query results")
        return _assessed_result(result, "failed", reason="reranker_failed")

    reranked = {
        **result,
        "distances": [[item[0] for item in ranked]],
        "documents": [[item[1] for item in ranked]],
        "metadatas": [[item[2] for item in ranked]],
    }
    if not ranked:
        return _assessed_result(reranked, "empty")
    return _assessed_result(reranked, "scored", best_score=max(item[0] for item in ranked))
