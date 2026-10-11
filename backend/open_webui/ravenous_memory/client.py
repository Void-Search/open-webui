"""Identity-scoped HTTP access; never read the memory service database."""

import os
from contextvars import ContextVar
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from ravenous_common.memory import MemoryClient

_scope = ContextVar('ravenous_memory_scope', default=None)


def configured():
    return os.environ.get('RAVENOUS_MEMORY_ENABLED') == 'true'


def begin_scope(metadata):
    return _scope.set(metadata)


def end_scope(token):
    _scope.reset(token)


async def scoped_stream(iterator, metadata):
    token = begin_scope(metadata)
    try:
        async for item in iterator:
            yield item
    finally:
        end_scope(token)


def enabled():
    metadata = _scope.get()
    return configured() and (metadata is None or metadata.get('ravenous_memory_owned', False))


def suppress_native(metadata):
    return configured() and bool(
        metadata.get('ravenous_memory_owned') or metadata.get('ravenous_memory_excluded')
    )


def client(user):
    key = os.environ.get('RAVENOUS_WEBUI_MEMORY_KEY')
    if not enabled() or not key or not user or not user.id:
        raise HTTPException(503, 'Memory identity is unavailable')
    return MemoryClient(os.environ.get('RAVENOUS_ORCHESTRATOR_BASE_URL', 'http://orchestrator:8000'), key, user.id)


async def call(user, method, path, body=None):
    try:
        return await client(user).request(method, path, body, timeout=10)
    except httpx.HTTPStatusError as error:
        raise HTTPException(error.response.status_code, 'Memory request failed') from error
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(503, 'Memory service unavailable') from error


async def rows(user, query=''):
    result = await call(user, 'GET', 'memories' + ('?query=' + quote(query, safe='') if query else ''))
    return [{**row, 'user_id': user.id} for row in result]


async def save(user, body, memory_id=None):
    suffix = 'memories' + ('/' + quote(memory_id, safe='') if memory_id else '')
    result = await call(user, 'PUT' if memory_id else 'POST', suffix, body)
    return {**result, 'user_id': user.id}


async def remove_session(user, chat_id):
    if enabled():
        await call(user, 'DELETE', 'sessions/' + quote(chat_id, safe=''))
