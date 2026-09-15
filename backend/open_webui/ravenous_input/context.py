"""Enforce local llama.cpp request budgets using its live tokenizer and context."""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

import httpx

from .settings import load_settings


class ContextBudgetError(RuntimeError):
    """A sanitized local-provider context failure."""


@dataclass(frozen=True)
class ContextBudgetResult:
    payload: dict[str, Any]
    context_tokens: int
    input_tokens: int
    output_tokens: int
    margin_tokens: int
    trimmed_turns: int


@dataclass
class QueryTaskContext:
    messages: list[dict[str, Any]]
    render: Callable[[list[dict[str, Any]]], Awaitable[str]]


_query_task_context: ContextVar[QueryTaskContext | None] = ContextVar('ravenous_query_task_context', default=None)


def set_query_task_context(
    messages: list[dict[str, Any]],
    render: Callable[[list[dict[str, Any]]], Awaitable[str]],
) -> Token:
    """Expose source history only to the current query-generation call."""
    return _query_task_context.set(QueryTaskContext(copy.deepcopy(messages), render))


def reset_query_task_context(token: Token) -> None:
    _query_task_context.reset(token)


def is_local_provider(base_url: str) -> bool:
    configured = os.getenv('RAVENOUS_LLAMA_CPP_BASE_URL', '').rstrip('/')
    return bool(configured) and base_url.rstrip('/') == configured


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContextBudgetError(f'local model {label} is invalid')
    return value


def _input_tokens(payload: Any) -> int:
    if not isinstance(payload, dict):
        raise ContextBudgetError('local model token count response is invalid')
    value = payload.get('input_tokens')
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextBudgetError('local model token count response is invalid')
    return value


def _removable_turn(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    user_indexes = [index for index, message in enumerate(messages) if message.get('role') == 'user']
    if len(user_indexes) < 2:
        return None
    latest_user = user_indexes[-1]
    for position, start in enumerate(user_indexes[:-1]):
        end = user_indexes[position + 1]
        if end > latest_user:
            break
        turn = messages[start:end]
        if all(message.get('role') not in {'system', 'developer'} for message in turn):
            return start, end
    return None


def _trim_oldest_turn(messages: list[dict[str, Any]]) -> bool:
    removable = _removable_turn(messages)
    if removable is None:
        return False
    start, end = removable
    del messages[start:end]
    return True


async def _json_response(response: Any, failure: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, Awaitable):
            payload = await payload
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ContextBudgetError(failure) from exc
    if not isinstance(payload, dict):
        raise ContextBudgetError(failure)
    return payload


@dataclass
class _BudgetRequest:
    client: Any
    headers: dict[str, str]
    api_base: str
    server_root: str
    outgoing: dict[str, Any]
    output_tokens: int
    margin: int


async def _enforce_budget(request: _BudgetRequest) -> ContextBudgetResult:
    try:
        props_response = await request.client.get(
            f'{request.server_root}/props',
            headers=request.headers,
            params={'model': request.outgoing.get('model', '')},
        )
        props = await _json_response(props_response, 'local model context discovery failed')
        generation = props.get('default_generation_settings')
        if not isinstance(generation, dict):
            raise ContextBudgetError('local model context discovery failed')
        context_tokens = _positive_integer(generation.get('n_ctx'), 'context limit')
        input_budget = context_tokens - request.output_tokens - request.margin
        if input_budget <= 0:
            raise ContextBudgetError('local model output reservation exhausts context')

        trimmed_turns = 0
        task_context = _query_task_context.get()
        while True:
            count_response = await request.client.post(
                f'{request.api_base}/chat/completions/input_tokens',
                headers=request.headers,
                json=request.outgoing,
            )
            count_payload = await _json_response(count_response, 'local model token counting failed')
            input_tokens = _input_tokens(count_payload)
            if input_tokens <= input_budget:
                return ContextBudgetResult(
                    payload=request.outgoing,
                    context_tokens=context_tokens,
                    input_tokens=input_tokens,
                    output_tokens=request.output_tokens,
                    margin_tokens=request.margin,
                    trimmed_turns=trimmed_turns,
                )

            if task_context is not None:
                if not _trim_oldest_turn(task_context.messages):
                    raise ContextBudgetError('local model protected content exceeds context')
                rendered = await task_context.render(copy.deepcopy(task_context.messages))
                messages = request.outgoing.get('messages')
                if (
                    not isinstance(rendered, str)
                    or not isinstance(messages, list)
                    or len(messages) != 1
                    or not isinstance(messages[0], dict)
                ):
                    raise ContextBudgetError('local query task could not be rerendered')
                messages[0]['content'] = rendered
            else:
                messages = request.outgoing.get('messages')
                if not isinstance(messages, list) or not _trim_oldest_turn(messages):
                    raise ContextBudgetError('local model protected content exceeds context')
            trimmed_turns += 1
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        raise ContextBudgetError('local model context validation failed') from exc


async def enforce_context_budget(
    payload: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    client: Any | None = None,
) -> ContextBudgetResult:
    """Count and trim an exact chat payload before sending it to llama.cpp."""
    settings = load_settings()
    outgoing = copy.deepcopy(payload)
    output_tokens = _positive_integer(outgoing.get('max_tokens', settings.default_output_tokens), 'output reservation')
    outgoing['max_tokens'] = output_tokens
    api_base = base_url.rstrip('/')
    server_root = api_base[:-3] if api_base.endswith('/v1') else api_base
    owns_client = client is None
    active_client = client or httpx.AsyncClient()
    request = _BudgetRequest(
        client=active_client,
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        api_base=api_base,
        server_root=server_root,
        outgoing=outgoing,
        output_tokens=output_tokens,
        margin=settings.context_margin_tokens,
    )
    try:
        return await asyncio.wait_for(
            _enforce_budget(request),
            timeout=settings.context_timeout_seconds,
        )
    except TimeoutError as exc:
        raise ContextBudgetError('local model context validation timed out') from exc
    finally:
        if owns_client:
            await active_client.aclose()
