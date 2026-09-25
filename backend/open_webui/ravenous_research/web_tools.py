"""Native chat adapter for the independent, identity-scoped web tools."""

import asyncio
import os

import httpx
from fastapi import HTTPException
from ravenous_common.auth import knowledge_auth_client

from .quality import initial_query as choose_query
from .quality import recovery_notice, requested_domains


def contextual_question(messages, question):
    """Resolve short follow-ups using bounded prior user requests, never model claims."""
    prior = [m['content'] for m in messages if m.get('role') == 'user' and isinstance(m.get('content'), str)]
    if prior and prior[-1] == question:
        prior.pop()
    if not prior:
        return (question or '')[:4096]
    # Keep the originating subject as well as recent clarifications. A chain of
    # "yes", "why did that fail?", "how can I do this?" must not erase it.
    selected = prior if len(prior) <= 8 else [*prior[:2], *prior[-6:]]
    context = '\n'.join(text[:400] for text in selected)
    return (
        'Current request: ' + (question or '')[:1800] + '\nEarlier user context (resolve references only):\n' + context
    )[:4096]


async def _post(user, tool, payload):
    if os.environ.get('RAVENOUS_RESEARCH_ENABLED') != 'true':
        raise HTTPException(503, 'Ravenous research is disabled')
    key = os.environ.get('RAVENOUS_WEBUI_RESEARCH_KEY')
    if not key or user is None or not user.id:
        raise HTTPException(403, 'Research caller identity is unavailable')
    endpoint = os.environ.get('RAVENOUS_ORCHESTRATOR_BASE_URL', 'http://orchestrator:8000')
    try:
        async with knowledge_auth_client(endpoint, key, user.id, timeout=140 if tool == 'research' else 40) as client:
            response = await client.post('/v1/tools/' + tool, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise HTTPException(status if status in (401, 403) else 503, 'Ravenous web tool request failed') from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, 'Ravenous web tools are unavailable') from exc


async def search(user, query, count=3):
    result = await _post(user, 'search', {'query': query, 'limit': max(1, min(count or 3, 3))})
    return [
        dict(link=item['source']['url'], title=item['source'].get('title'), snippet=item['snippet'])
        for item in result['results']
    ]


async def fetch(user, url):
    return await _post(user, 'fetch', {'url': url, 'timeout_seconds': 30})


async def research(user, query, count=3, domain_filters=None, initial_query=None, source_domains=None):
    payload = {
        'query': query,
        'limit': max(1, min(count or 3, 3)),
        'max_attempts': 2,
        'domain_filters': domain_filters or [],
    }
    domains = source_domains if source_domains is not None else requested_domains(query)
    if domains:
        payload['requested_domains'] = domains
    if initial_query:
        payload['initial_query'] = initial_query
    return await _post(user, 'research', payload)


async def shared_research(request, user, query, count=3, domain_filters=None, initial_query=None, source_domains=None):
    if request is None:
        return await research(user, query, count, domain_filters, initial_query, source_domains)
    task = getattr(request.state, 'ravenous_research_task', None)
    if task is None:
        task = asyncio.create_task(research(user, query, count, domain_filters, initial_query, source_domains))
        request.state.ravenous_research_task = task
    return await task


def assessment_notice(result):
    return recovery_notice(result)


def evidence_documents(result):
    passages = {}
    for passage in result.get('supporting_passages', []):
        passages.setdefault(passage['url'], []).append(passage['text'])
    docs = []
    for item in result.get('evidence', []):
        url = item['source']['url']
        # An unverified page is not enough for a partial answer. Retain the raw
        # API evidence for diagnostics, but expose only validated passages here.
        text = (
            item['text']
            if result.get('sufficient')
            else '\n\n'.join(passage for passage in passages.get(url, []) if passage in item['text'])
        )
        if not text.strip():
            continue
        docs.append(
            {
                'content': text,
                'metadata': {
                    'source': url,
                    'link': url,
                    'title': item['source'].get('title'),
                    'fetched_at': item['fetched_at'],
                    'content_sha256': item['content_sha256'],
                    'research_sufficient': result['sufficient'],
                },
            }
        )
    return docs


async def process(user, queries, count=3, domain_filters=None, question=None, source_domains=None, request=None):
    if not queries:
        raise HTTPException(422, 'A search query is required')
    question = question or queries[0]
    result = await shared_research(
        request, user, question, count, domain_filters, choose_query(queries, question), source_domains
    )
    notice = assessment_notice(result)
    docs = evidence_documents(result)
    urls = [doc['metadata']['source'] for doc in docs]
    return {
        'status': True,
        'collection_name': None,
        'filenames': urls,
        'items': [
            {'link': doc['metadata']['source'], 'title': doc['metadata']['title'], 'snippet': doc['content']}
            for doc in docs
        ],
        'docs': docs,
        'loaded_count': len(docs),
        'research': result,
        'research_notice': notice,
    }
