"""Tests for live llama.cpp context budgeting."""

import asyncio

import pytest
from open_webui.ravenous_input.context import (
    ContextBudgetError,
    enforce_context_budget,
    is_local_provider,
    reset_query_task_context,
    set_query_task_context,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, context_tokens, counts):
        self.context_tokens = context_tokens
        self.counts = iter(counts)
        self.requests = []

    async def get(self, url, **kwargs):
        self.requests.append(('GET', url, kwargs))
        return FakeResponse(
            {
                'default_generation_settings': {'n_ctx': self.context_tokens},
                'total_slots': 4,
            }
        )

    async def post(self, url, **kwargs):
        self.requests.append(('POST', url, kwargs))
        return FakeResponse({'input_tokens': next(self.counts)})


def configure(monkeypatch, *, output=20, margin=5):
    monkeypatch.setenv('RAVENOUS_INPUT_DEFAULT_OUTPUT_TOKENS', str(output))
    monkeypatch.setenv('RAVENOUS_INPUT_CONTEXT_MARGIN_TOKENS', str(margin))
    monkeypatch.setenv('RAVENOUS_INPUT_CONTEXT_TIMEOUT_SECONDS', '1')


def run_budget(payload, client):
    return asyncio.run(
        enforce_context_budget(
            payload,
            base_url='http://llama.test/v1',
            api_key='secret',
            client=client,
        )
    )


def test_uses_live_per_request_context_without_dividing_parallel_slots(monkeypatch):
    configure(monkeypatch)
    client = FakeClient(100, [75])

    result = run_budget(
        {'model': 'model-a', 'messages': [{'role': 'user', 'content': 'latest'}]},
        client,
    )

    assert result.context_tokens == 100
    assert result.input_tokens == 75
    assert result.payload['max_tokens'] == 20
    assert client.requests[0][2]['params'] == {'model': 'model-a'}
    assert client.requests[1][1].endswith('/v1/chat/completions/input_tokens')


def test_trims_oldest_complete_turn_including_tool_pair(monkeypatch):
    configure(monkeypatch)
    client = FakeClient(100, [90, 30])
    messages = [
        {'role': 'system', 'content': 'instructions'},
        {'role': 'user', 'content': 'old'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1'}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'latest'},
    ]

    result = run_budget({'model': 'model-a', 'messages': messages}, client)

    assert result.trimmed_turns == 1
    assert result.payload['messages'] == [messages[0], messages[-1]]
    assert messages[1]['content'] == 'old'


def test_rerenders_query_task_after_trimming_source_history(monkeypatch):
    configure(monkeypatch)
    client = FakeClient(100, [90, 40])
    history = [
        {'role': 'user', 'content': 'old'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'latest'},
    ]

    async def render(messages):
        return ' | '.join(message['content'] for message in messages)

    token = set_query_task_context(history, render)
    try:
        result = run_budget(
            {'model': 'model-a', 'messages': [{'role': 'user', 'content': 'rendered'}]},
            client,
        )
    finally:
        reset_query_task_context(token)

    assert result.payload['messages'][0]['content'] == 'latest'
    assert history[0]['content'] == 'old'


@pytest.mark.parametrize(
    ('context_tokens', 'output', 'margin'),
    [(64, 64, 0), (64, 63, 1), (0, 10, 1)],
)
def test_rejects_invalid_or_exhausted_context(monkeypatch, context_tokens, output, margin):
    configure(monkeypatch, output=output, margin=margin)
    client = FakeClient(context_tokens, [])

    with pytest.raises(ContextBudgetError):
        run_budget(
            {'model': 'model-a', 'messages': [{'role': 'user', 'content': 'latest'}]},
            client,
        )


def test_rejects_oversized_protected_content(monkeypatch):
    configure(monkeypatch)
    client = FakeClient(100, [76])

    with pytest.raises(ContextBudgetError, match='protected content'):
        run_budget(
            {
                'model': 'model-a',
                'messages': [
                    {'role': 'system', 'content': 'instructions'},
                    {'role': 'user', 'content': 'latest'},
                ],
            },
            client,
        )


def test_live_context_is_queried_for_every_call(monkeypatch):
    configure(monkeypatch)
    payload = {'model': 'model-a', 'messages': [{'role': 'user', 'content': 'latest'}]}

    first = run_budget(payload, FakeClient(100, [20]))
    second = run_budget(payload, FakeClient(200, [20]))

    assert first.context_tokens == 100
    assert second.context_tokens == 200


def test_local_provider_requires_exact_configured_base(monkeypatch):
    monkeypatch.setenv('RAVENOUS_LLAMA_CPP_BASE_URL', 'http://llama.test/v1')

    assert is_local_provider('http://llama.test/v1/') is True
    assert is_local_provider('http://other.test/v1') is False
