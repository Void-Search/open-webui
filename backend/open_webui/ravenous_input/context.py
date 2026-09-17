"""Adapt request-local query history to shared context-budget enforcement."""

import asyncio
import copy
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

import httpx
from ravenous_common import context as common_context
from ravenous_common.context import ContextBudgetError, ContextBudgetResult, is_local_provider
from ravenous_common.settings import load_settings

__all__ = [
    'ContextBudgetError',
    'ContextBudgetResult',
    'is_local_provider',
    'enforce_context_budget',
    'set_query_task_context',
    'reset_query_task_context',
]


@dataclass
class QueryTaskContext:
    messages: list[dict[str, Any]]
    render: Callable[[list[dict[str, Any]]], Awaitable[str]]


_query_task_context: ContextVar[QueryTaskContext | None] = ContextVar(
    'ravenous_query_task_context',
    default=None,
)


def set_query_task_context(messages, render) -> Token:
    return _query_task_context.set(QueryTaskContext(copy.deepcopy(messages), render))


def reset_query_task_context(token: Token) -> None:
    _query_task_context.reset(token)


async def enforce_context_budget(payload, *, base_url, api_key, client=None) -> ContextBudgetResult:
    settings = load_settings()
    task = _query_task_context.get()
    active_client = client if client is not None else httpx.AsyncClient(trust_env=False)
    try:
        return await asyncio.wait_for(
            common_context.enforce_context_budget(
                payload,
                base_url=base_url,
                api_key=api_key,
                client=active_client,
                settings=settings,
                history=task.messages if task else None,
                render_history=task.render if task else None,
            ),
            timeout=settings.context_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ContextBudgetError('local model context validation timed out') from exc
    finally:
        if client is None:
            await active_client.aclose()
