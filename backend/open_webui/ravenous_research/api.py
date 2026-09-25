"""Session-authenticated gateway carrying an operator-mapped stable caller identity."""

import os
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from ravenous_common.auth import knowledge_auth_client

from open_webui.utils.auth import get_verified_user

router = APIRouter()
RESOURCE = r"[a-zA-Z0-9_-]+"
KNOWLEDGE_PATH = re.compile(
    rf"(?:documents(?:/{RESOURCE}(?:/(?:status|download|shares(?:/{RESOURCE})?|"
    rf"revisions/{RESOURCE}/pin))?)?|collections(?:/{RESOURCE}(?:/"
    rf"(?:documents/{RESOURCE}|shares(?:/{RESOURCE})?))?)?|retrieve)"
)


def enabled():
    return os.environ.get('RAVENOUS_RESEARCH_ENABLED', 'false') == 'true'


@router.get('/config')
async def configuration(_user=Depends(get_verified_user)):
    return {'enabled': enabled()}


async def forward(request, user, endpoint, path):
    if not enabled():
        raise HTTPException(404, 'Independent research is disabled')
    key = os.environ.get('RAVENOUS_WEBUI_RESEARCH_KEY')
    if not key:
        raise HTTPException(503, 'Research identity is not configured')
    headers = {name: request.headers[name] for name in (
        'content-type', 'idempotency-key', 'x-filename'
    ) if name in request.headers}
    # The authenticated session supplies the delegated subject, never browser input.
    try:
        async with knowledge_auth_client(endpoint, key, user.id, timeout=190) as client:
            response = await client.request(request.method, '/v1/' + path,
                params=request.query_params, headers=headers, content=request.stream())
        return Response(response.content, status_code=response.status_code,
                        media_type=response.headers.get('content-type', 'application/json'),
                        headers={'Cache-Control': 'no-store'})
    except httpx.HTTPError as exc:
        raise HTTPException(503, 'Independent research is unavailable') from exc


@router.api_route('/knowledge/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
async def knowledge(path: str, request: Request, user=Depends(get_verified_user)):
    if not KNOWLEDGE_PATH.fullmatch(path):
        raise HTTPException(404, 'Research resource not found')
    return await forward(request, user,
        os.environ.get('RAVENOUS_KNOWLEDGE_BASE_URL', 'http://knowledge:8000'), path)


@router.post('/chat')
async def chat(request: Request, user=Depends(get_verified_user)):
    # Dedicated route never enters native retrieval, web fetching or title/tag tasks.
    return await forward(request, user,
        os.environ.get('RAVENOUS_ORCHESTRATOR_BASE_URL', 'http://orchestrator:8000'),
        'chat/completions')
