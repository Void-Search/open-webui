"""Request-scoped reuse of canonical shared prompt preparation."""

import asyncio
from typing import Any

from ravenous_common.pipeline import PreparationResult, prepare_prompt

_TASK_ATTRIBUTE = '_ravenous_prompt_preparation_task'

__all__ = ['PreparationResult', 'prepare_prompt', 'prepare_request_prompt']


async def prepare_request_prompt(request: Any, raw: str) -> PreparationResult:
    task = getattr(request.state, _TASK_ATTRIBUTE, None)
    if task is None:
        task = asyncio.create_task(prepare_prompt(raw))
        setattr(request.state, _TASK_ATTRIBUTE, task)
    return await asyncio.shield(task)
