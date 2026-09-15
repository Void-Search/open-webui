"""Bounded LanguageTool HTTP correction for one prompt turn."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from .protected import ProtectedText, protect_text
from .settings import InputSettings, load_settings

_APPROVED_CATEGORIES = {'GRAMMAR', 'TYPOS'}
_APPROVED_ISSUE_TYPES = {'grammar', 'misspelling'}
_client: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None
_settings: InputSettings | None = None


@dataclass(frozen=True)
class CorrectionResult:
    text: str
    status: str
    reason: str | None
    accepted_matches: int
    skipped_matches: int
    milliseconds: float


def _result(
    text: str,
    started: float,
    *,
    status: str,
    reason: str | None = None,
    accepted: int = 0,
    skipped: int = 0,
) -> CorrectionResult:
    return CorrectionResult(
        text=text,
        status=status,
        reason=reason,
        accepted_matches=accepted,
        skipped_matches=skipped,
        milliseconds=round((time.monotonic() - started) * 1000, 3),
    )


async def start_checker() -> None:
    """Wait for the configured service to accept a real grammar request."""
    global _client, _semaphore, _settings  # pylint: disable=global-statement
    if _client is not None:
        return
    settings = load_settings()
    _settings = settings
    _semaphore = asyncio.Semaphore(settings.correction_concurrency)
    if not settings.correction_enabled:
        return

    base_url = os.getenv('RAVENOUS_LANGUAGETOOL_BASE_URL', 'http://languagetool:8081').rstrip('/')
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError('LanguageTool base URL must be an HTTP(S) service URL')
    try:
        startup_timeout = float(os.getenv('RAVENOUS_LANGUAGETOOL_STARTUP_TIMEOUT_SECONDS', '60'))
    except ValueError as exc:
        raise RuntimeError('LanguageTool startup timeout must be numeric') from exc
    if not 0 < startup_timeout <= 3600:
        raise RuntimeError('LanguageTool startup timeout must be between 0 and 3600 seconds')
    _client = httpx.AsyncClient(base_url=base_url, trust_env=False)
    deadline = time.monotonic() + startup_timeout
    try:
        while time.monotonic() < deadline:
            try:
                remaining = max(0.001, deadline - time.monotonic())
                response = await asyncio.wait_for(
                    _client.post(
                        '/v2/check',
                        data={'language': settings.language, 'text': 'Warm checker.'},
                        timeout=min(10, remaining),
                    ),
                    timeout=remaining,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get('matches'), list):
                    return
            except (TimeoutError, httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise RuntimeError('LanguageTool did not become ready')
    except BaseException:
        await stop_checker()
        raise


async def stop_checker() -> None:
    """Close our HTTP client; the service has an independent lifecycle."""
    global _client, _semaphore, _settings  # pylint: disable=global-statement
    client = _client
    _client = None
    _semaphore = None
    _settings = None
    if client is not None:
        await client.aclose()


def utf16_offset_to_index(text: str, offset: int) -> int:
    """Convert a LanguageTool UTF-16 code-unit offset to a Python index."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('invalid UTF-16 offset')
    units = 0
    for index, character in enumerate(text):
        if units == offset:
            return index
        next_units = units + len(character.encode('utf-16-le')) // 2
        if offset < next_units:
            raise ValueError('UTF-16 offset splits a code point')
        units = next_units
    if units == offset:
        return len(text)
    raise ValueError('UTF-16 offset is outside the text')


def _approved(match: dict[str, Any]) -> bool:
    rule = match.get('rule')
    if not isinstance(rule, dict):
        return False
    category = rule.get('category')
    category_id = category.get('id') if isinstance(category, dict) else None
    issue_type = rule.get('issueType')
    return category_id in _APPROVED_CATEGORIES or issue_type in _APPROVED_ISSUE_TYPES


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _candidate(protected: ProtectedText, match: Any) -> tuple[int, int, str] | None:
    if not isinstance(match, dict) or not _approved(match):
        return None
    replacements = match.get('replacements')
    replacement = replacements[0] if isinstance(replacements, list) and len(replacements) == 1 else None
    replacement = replacement.get('value') if isinstance(replacement, dict) else None
    offset, length = match.get('offset'), match.get('length')
    if not isinstance(replacement, str) or not replacement or not _is_index(offset) or not _is_index(length):
        return None
    try:
        start = utf16_offset_to_index(protected.masked, offset)
        end = utf16_offset_to_index(protected.masked, offset + length)
    except ValueError:
        return None
    if protected.overlaps(start, end) or protected.masked[start:end] == replacement:
        return None
    return start, end, replacement


def _overlapping_indexes(candidates: list[tuple[int, int, str]]) -> set[int]:
    overlapping = set()
    for left_index, (left_start, left_end, _) in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right_start, right_end, _ = candidates[right_index]
            if right_start > left_end:
                break
            if max(left_start, right_start) < min(left_end, right_end) or (
                left_start == left_end == right_start == right_end
            ):
                overlapping.update((left_index, right_index))
    return overlapping


def _apply_matches(protected: ProtectedText, matches: list[Any]) -> tuple[str | None, int, int]:
    candidates = [candidate for match in matches if (candidate := _candidate(protected, match))]
    candidates.sort(key=lambda item: (item[0], item[1]))
    overlapping = _overlapping_indexes(candidates)
    accepted = [item for index, item in enumerate(candidates) if index not in overlapping]
    corrected = protected.masked
    for start, end, replacement in reversed(accepted):
        corrected = f'{corrected[:start]}{replacement}{corrected[end:]}'
    restored = protected.restore(corrected)
    skipped = len(matches) - len(accepted)
    return restored, len(accepted), skipped


def _unavailable_reason(text: str, settings: InputSettings) -> str | None:
    if not settings.correction_enabled:
        return 'disabled'
    if len(text) > settings.correction_max_characters:
        return 'overlong'
    if _client is None or _semaphore is None:
        return 'unavailable'
    if _semaphore.locked():
        return 'overload'
    return None


async def correct_spelling_grammar(text: str) -> CorrectionResult:
    """Correct approved matches or conservatively skip the correction stage."""
    started = time.monotonic()
    settings = _settings or load_settings()
    reason = _unavailable_reason(text, settings)
    if reason is not None:
        return _result(text, started, status='skipped', reason=reason)

    protected = protect_text(text)

    async def request_matches() -> list[Any]:
        async with _semaphore:
            response = await _client.post(
                '/v2/check',
                data={'language': settings.language, 'text': protected.masked},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get('matches'), list):
                raise ValueError('invalid LanguageTool response')
            return payload['matches']

    try:
        matches = await asyncio.wait_for(request_matches(), timeout=settings.correction_timeout_ms / 1000)
        corrected, accepted, skipped = _apply_matches(protected, matches)
        if corrected is None:
            return _result(text, started, status='skipped', reason='unavailable', skipped=len(matches))
        return _result(
            corrected,
            started,
            status='corrected' if corrected != text else 'unchanged',
            accepted=accepted,
            skipped=skipped,
        )
    except (TimeoutError, httpx.TimeoutException):
        return _result(text, started, status='skipped', reason='timeout')
    except (httpx.HTTPError, ValueError, TypeError):
        return _result(text, started, status='skipped', reason='unavailable')
