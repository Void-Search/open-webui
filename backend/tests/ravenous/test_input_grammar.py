"""Contract tests for local LanguageTool correction filtering."""

import asyncio

from open_webui.ravenous_input.settings import InputSettings
from ravenous_common import grammar


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:  # pylint: disable=too-few-public-methods
    def __init__(self, responder):
        self.responder = responder
        self.requests = []

    async def post(self, path, **kwargs):
        self.requests.append((path, kwargs))
        return await self.responder(path, kwargs)


def settings(**overrides):
    values = {
        'correction_enabled': True,
        'language': 'en-AU',
        'correction_timeout_ms': 200,
        'correction_max_characters': 2000,
        'correction_concurrency': 2,
        'cleanup_profile': 'conservative',
        'default_output_tokens': 2048,
        'context_margin_tokens': 64,
        'context_timeout_seconds': 10,
    }
    values.update(overrides)
    return InputSettings(**values)


def configure(monkeypatch, responder, **overrides):
    client = FakeClient(responder)
    monkeypatch.setattr(grammar, '_client', client)
    monkeypatch.setattr(grammar, '_settings', settings(**overrides))
    monkeypatch.setattr(grammar, '_semaphore', asyncio.Semaphore(2))
    return client


def match(text, needle, replacement, *, replacements=None, category='TYPOS'):
    offset = len(text[: text.index(needle)].encode('utf-16-le')) // 2
    length = len(needle.encode('utf-16-le')) // 2
    return {
        'offset': offset,
        'length': length,
        'replacements': replacements or [{'value': replacement}],
        'rule': {'category': {'id': category}, 'issueType': 'misspelling'},
    }


def test_applies_one_unambiguous_spelling_match(monkeypatch):
    async def responder(_path, kwargs):
        text = kwargs['data']['text']
        return FakeResponse({'matches': [match(text, 'teh', 'the')]})

    configure(monkeypatch, responder)
    result = asyncio.run(grammar.correct_spelling_grammar('Fix teh typo.'))

    assert result.text == 'Fix the typo.'
    assert result.status == 'corrected'
    assert result.accepted_matches == 1


def test_protected_code_url_emoji_and_negation_are_unchanged(monkeypatch):
    async def responder(_path, kwargs):
        text = kwargs['data']['text']
        return FakeResponse({'matches': [match(text, 'teh', 'the')]})

    configure(monkeypatch, responder)
    source = 'Fix teh but not `teh` at https://teh.test 😀.'
    result = asyncio.run(grammar.correct_spelling_grammar(source))

    assert result.text == 'Fix the but not `teh` at https://teh.test 😀.'


def test_ambiguous_and_style_matches_are_skipped(monkeypatch):
    async def responder(_path, kwargs):
        text = kwargs['data']['text']
        ambiguous = match(
            text,
            'word',
            'ward',
            replacements=[{'value': 'ward'}, {'value': 'world'}],
        )
        style = match(text, 'plain', 'simple', category='STYLE')
        style['rule']['issueType'] = 'style'
        return FakeResponse({'matches': [ambiguous, style]})

    configure(monkeypatch, responder)
    result = asyncio.run(grammar.correct_spelling_grammar('word plain'))

    assert result.text == 'word plain'
    assert result.status == 'unchanged'
    assert result.skipped_matches == 2


def test_utf16_offsets_account_for_emoji_surrogate_pairs():
    text = 'A😀teh'

    assert grammar.utf16_offset_to_index(text, 3) == 2


def test_timeout_and_overload_fail_open(monkeypatch):
    async def blocked(_path, _kwargs):
        await asyncio.Event().wait()

    configure(monkeypatch, blocked, correction_timeout_ms=1)
    timeout = asyncio.run(grammar.correct_spelling_grammar('Fix teh.'))
    assert timeout.text == 'Fix teh.'
    assert timeout.reason == 'timeout'

    async def empty(_path, _kwargs):
        return FakeResponse({'matches': []})

    configure(monkeypatch, empty)
    semaphore = asyncio.Semaphore(1)
    asyncio.run(semaphore.acquire())
    monkeypatch.setattr(grammar, '_semaphore', semaphore)
    overloaded = asyncio.run(grammar.correct_spelling_grammar('Fix teh.'))
    assert overloaded.reason == 'overload'


def test_disabled_overlong_and_malformed_responses_fail_open(monkeypatch):
    async def malformed(_path, _kwargs):
        return FakeResponse({'unexpected': []})

    configure(monkeypatch, malformed, correction_enabled=False)
    assert asyncio.run(grammar.correct_spelling_grammar('text')).reason == 'disabled'

    configure(monkeypatch, malformed, correction_max_characters=3)
    assert asyncio.run(grammar.correct_spelling_grammar('long')).reason == 'overlong'

    configure(monkeypatch, malformed)
    assert asyncio.run(grammar.correct_spelling_grammar('text')).reason == 'unavailable'
