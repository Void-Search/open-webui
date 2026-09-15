"""Request-scoped prompt correction and conservative cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

from .cleanup import clean_prompt
from .grammar import correct_spelling_grammar

_TASK_ATTRIBUTE = '_ravenous_prompt_preparation_task'


@dataclass(frozen=True)
class PreparationResult:  # pylint: disable=too-many-instance-attributes
    raw: str
    prepared: str
    changed: bool
    correction_status: str
    correction_reason: str | None
    accepted_matches: int
    skipped_matches: int
    correction_ms: float
    cleanup_ms: float
    total_ms: float

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop('raw')
        values.update(
            {
                'raw_sha256': hashlib.sha256(self.raw.encode()).hexdigest(),
                'prepared_sha256': hashlib.sha256(self.prepared.encode()).hexdigest(),
                'raw_characters': len(self.raw),
                'prepared_characters': len(self.prepared),
            }
        )
        return values


async def prepare_prompt(raw: str, *, plain_text: bool = False) -> PreparationResult:
    """Correct one raw turn, then clean it regardless of checker availability."""
    if not isinstance(raw, str):
        raise TypeError('prompt must be text')
    started = time.monotonic()
    correction = await correct_spelling_grammar(raw)
    cleanup_started = time.monotonic()
    prepared = clean_prompt(correction.text, plain_text=plain_text)
    cleanup_ms = round((time.monotonic() - cleanup_started) * 1000, 3)
    return PreparationResult(
        raw=raw,
        prepared=prepared,
        changed=prepared != raw,
        correction_status=correction.status,
        correction_reason=correction.reason,
        accepted_matches=correction.accepted_matches,
        skipped_matches=correction.skipped_matches,
        correction_ms=correction.milliseconds,
        cleanup_ms=cleanup_ms,
        total_ms=round((time.monotonic() - started) * 1000, 3),
    )


async def prepare_request_prompt(request: Any, raw: str) -> PreparationResult:
    """Create one preparation task per request and share it across model fanout."""
    task = getattr(request.state, _TASK_ATTRIBUTE, None)
    if task is None:
        task = asyncio.create_task(prepare_prompt(raw))
        setattr(request.state, _TASK_ATTRIBUTE, task)
    return await asyncio.shield(task)
