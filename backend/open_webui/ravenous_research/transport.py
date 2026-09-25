"""Private identity-scoped transport; acquisition progress never contains credentials."""

import asyncio
import json
import os
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from ravenous_common.auth import knowledge_auth_client


def identity(user):
    key = os.environ.get('RAVENOUS_WEBUI_RESEARCH_KEY')
    if os.environ.get('RAVENOUS_RESEARCH_ENABLED') != 'true':
        raise HTTPException(503, 'Research is disabled')
    if not key or not user or not user.id:
        raise HTTPException(403, 'Research identity is unavailable')
    return key


async def batch(user, payload, emit):
    key = identity(user)
    endpoint = os.environ.get('RAVENOUS_ORCHESTRATOR_BASE_URL', 'http://orchestrator:8000')
    async with knowledge_auth_client(endpoint, key, user.id, timeout=140) as client:
        async with client.stream('POST', '/v1/tools/research/batch', json={**payload, 'stream': True}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data: '):
                    continue
                event = json.loads(line[6:])
                if event['type'] == 'result':
                    return event['data']
                if event['type'] == 'error':
                    code = event['data']['code']
                    raise HTTPException(403 if code in ('authorization_denied', 'web_disabled') else 503, code)
                if event['type'] in ('query', 'page'):
                    await emit(event)
    raise httpx.ReadError('Incomplete research stream')


async def saved(user, question, queries):
    key = identity(user)
    endpoint = os.environ.get('RAVENOUS_KNOWLEDGE_BASE_URL', 'http://knowledge:8000')
    async with knowledge_auth_client(endpoint, key, user.id, timeout=35) as client:
        response = await client.post(
            '/v1/retrieve',
            json={
                'queries': [{'text': query} for query in queries],
                'resolved_intent': question,
                'include_saved_research': True,
                'limit': 75,
                'candidate_only': True,
            },
        )
        response.raise_for_status()
    return response.json()['hits']


async def authorized_revisions(user, references):
    """Recheck saved-source access and revision identity immediately before use."""
    if not references:
        return set()
    key = identity(user)
    endpoint = os.environ.get('RAVENOUS_KNOWLEDGE_BASE_URL', 'http://knowledge:8000')
    slots = asyncio.Semaphore(8)
    async with knowledge_auth_client(endpoint, key, user.id, timeout=8) as client:

        async def check(document, revision):
            async with slots:
                response = await client.get('/v1/documents/' + quote(document, safe=''))
                if response.status_code in (403, 404):
                    return None
                response.raise_for_status()
                return (document, revision) if response.json().get('revision_id') == revision else None

        tasks = [asyncio.create_task(check(*reference)) for reference in references]
        try:
            return {value for value in await asyncio.gather(*tasks) if value}
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
