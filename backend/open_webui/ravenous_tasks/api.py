"""Fixed task routes; browser credentials and subjects never reach the services."""

import os
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from open_webui.utils.auth import get_verified_user

router = APIRouter()
TASK_PATH = re.compile(
    r'tasks(?:/task_[a-f0-9]{32}(?:/(?:events|artifacts(?:/[a-f0-9]{64})?|resolve|resume|cancel))?)?'
)


@router.get('/config')
async def configuration(_user=Depends(get_verified_user)):
    return {'enabled': os.environ.get('RAVENOUS_ADAPTIVE_ENABLED') == 'true',
            'execution': os.environ.get('RAVENOUS_ADAPTIVE_EXECUTION') == 'true'}


@router.get('/sources')
async def sources(user=Depends(get_verified_user)):
    if os.environ.get('RAVENOUS_ADAPTIVE_ENABLED') != 'true':
        raise HTTPException(404, 'Adaptive tasks are unavailable')
    key = os.environ.get('RAVENOUS_WEBUI_TASK_KEY')
    if not key:
        raise HTTPException(503, 'Task identity is not configured')
    async with httpx.AsyncClient(
        base_url=os.environ.get('RAVENOUS_TASK_KNOWLEDGE_BASE_URL', 'http://knowledge:8000'),
        headers={'Authorization': 'Bearer ' + key, 'X-Ravenous-Delegated-Subject': user.id},
        timeout=10, trust_env=False, follow_redirects=False,
    ) as client:
        response = await client.get('/v1/documents', params={'limit': 50})
        if response.status_code in (401, 403):
            raise HTTPException(response.status_code, 'Knowledge access is unavailable')
        response.raise_for_status()
        return response.json()


@router.get('/capabilities')
async def capabilities(user=Depends(get_verified_user)):
    if os.environ.get('RAVENOUS_ADAPTIVE_ENABLED') != 'true':
        raise HTTPException(404, 'Adaptive tasks are unavailable')
    key = os.environ.get('RAVENOUS_WEBUI_TASK_KEY')
    if not key:
        raise HTTPException(503, 'Task identity is not configured')
    try:
        async with httpx.AsyncClient(
            base_url=os.environ.get('RAVENOUS_TASK_BASE_URL', 'http://orchestrator:8000'),
            headers={'Authorization': 'Bearer ' + key, 'X-Ravenous-Delegated-Subject': user.id},
            timeout=10, trust_env=False, follow_redirects=False,
        ) as client:
            response = await client.get('/v1/tasks/capabilities')
    except httpx.HTTPError as exc:
        raise HTTPException(503, 'Task service is unavailable') from exc
    return Response(response.content, status_code=response.status_code,
                    media_type=response.headers.get('content-type', 'application/json'),
                    headers={'Cache-Control': 'no-store'})


@router.api_route('/{path:path}', methods=['GET', 'POST'])
async def forward(path: str, request: Request, user=Depends(get_verified_user)):
    if os.environ.get('RAVENOUS_ADAPTIVE_ENABLED') != 'true' or not TASK_PATH.fullmatch(path):
        raise HTTPException(404, 'Adaptive tasks are unavailable')
    key = os.environ.get('RAVENOUS_WEBUI_TASK_KEY')
    if not key:
        raise HTTPException(503, 'Task identity is not configured')
    headers = {'Authorization': 'Bearer ' + key, 'X-Ravenous-Delegated-Subject': user.id}
    for name in ('content-type', 'idempotency-key', 'last-event-id'):
        if name in request.headers:
            headers[name] = request.headers[name]
    client = httpx.AsyncClient(
        base_url=os.environ.get('RAVENOUS_TASK_BASE_URL', 'http://orchestrator:8000'),
        headers=headers,
        timeout=httpx.Timeout(30, read=None if path.endswith('/events') else 30),
        trust_env=False, follow_redirects=False,
    )
    try:
        outgoing = client.build_request(request.method, '/v1/' + path, content=await request.body())
        response = await client.send(outgoing, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(503, 'Task service is unavailable') from exc

    async def close():
        await response.aclose()
        await client.aclose()

    if not path.endswith('/events') or response.status_code != 200:
        try:
            return Response(await response.aread(), status_code=response.status_code,
                            media_type=response.headers.get('content-type', 'application/json'),
                            headers={'Cache-Control': 'no-store'})
        finally:
            await close()

    async def events():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await close()

    # Stream disconnect closes this transport only; explicit cancel owns the task.
    return StreamingResponse(events(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
    })
