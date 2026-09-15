"""Tests for preparation fallback and request-scoped reuse."""

import asyncio
from types import SimpleNamespace

from open_webui.ravenous_input import pipeline
from open_webui.ravenous_input.grammar import CorrectionResult


def correction(text, *, status='unchanged', reason=None):
    return CorrectionResult(
        text=text,
        status=status,
        reason=reason,
        accepted_matches=0,
        skipped_matches=0,
        milliseconds=1.25,
    )


def test_skipped_correction_still_cleans_original(monkeypatch):
    async def skipped(text):
        return correction(text, status='skipped', reason='timeout')

    monkeypatch.setattr(pipeline, 'correct_spelling_grammar', skipped)
    result = asyncio.run(pipeline.prepare_prompt('Hello   world'))

    assert result.raw == 'Hello   world'
    assert result.prepared == 'Hello world'
    assert result.changed is True
    assert result.correction_reason == 'timeout'
    assert result.public_dict()['raw_sha256'] != result.public_dict()['prepared_sha256']


def test_request_fanout_shares_exactly_one_preparation_task(monkeypatch):
    calls = 0
    release = asyncio.Event()

    async def prepare(raw, *, plain_text=False):
        nonlocal calls
        del plain_text
        calls += 1
        await release.wait()
        return raw

    monkeypatch.setattr(pipeline, 'prepare_prompt', prepare)

    async def exercise():
        request = SimpleNamespace(state=SimpleNamespace())
        first = asyncio.create_task(pipeline.prepare_request_prompt(request, 'raw'))
        second = asyncio.create_task(pipeline.prepare_request_prompt(request, 'prepared'))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second)

    results = asyncio.run(exercise())

    assert calls == 1
    assert results == ['raw', 'raw']
