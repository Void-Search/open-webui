"""Service readiness, remote addressing, outages, and HTTP-client ownership."""

import asyncio

import httpx
import pytest
from ravenous_common import grammar


@pytest.fixture(autouse=True)
def checker_state(monkeypatch):
    monkeypatch.setattr(grammar, '_client', None)
    monkeypatch.setattr(grammar, '_settings', None)
    monkeypatch.setattr(grammar, '_semaphore', None)
    monkeypatch.setenv('RAVENOUS_INPUT_CORRECTION_ENABLED', 'true')
    monkeypatch.setenv('RAVENOUS_INPUT_LANGUAGE', 'en-AU')
    monkeypatch.setenv('RAVENOUS_LANGUAGETOOL_STARTUP_TIMEOUT_SECONDS', '1')


def mock_client(monkeypatch, handler):
    clients = []
    client_type = httpx.AsyncClient

    def factory(**kwargs):
        client = client_type(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(grammar.httpx, 'AsyncClient', factory)
    return clients


def test_readiness_checks_remote_endpoint_with_configured_language(monkeypatch):
    requests = []
    monkeypatch.setenv('RAVENOUS_LANGUAGETOOL_BASE_URL', 'https://grammar.example.test/tool')

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'matches': []})

    clients = mock_client(monkeypatch, handler)

    async def exercise():
        await grammar.start_checker()
        await grammar.start_checker()
        assert len(clients) == 1
        assert grammar._client is clients[0]
        await grammar.stop_checker()

    asyncio.run(exercise())
    assert str(requests[0].url) == 'https://grammar.example.test/tool/v2/check'
    assert requests[0].content == b'language=en-AU&text=Warm+checker.'
    assert clients[0].is_closed
    assert grammar._client is None


def test_readiness_retries_invalid_and_unavailable_responses(monkeypatch):
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(200, json={'unexpected': []}),
            httpx.Response(200, json={'matches': []}),
        ]
    )
    clients = mock_client(monkeypatch, lambda _request: next(responses))

    async def exercise():
        await grammar.start_checker()
        assert grammar._client is not None
        await grammar.stop_checker()

    asyncio.run(exercise())
    assert clients[0].is_closed
    assert next(responses, None) is None


def test_readiness_failure_closes_client_and_blocks_startup(monkeypatch):
    monkeypatch.setenv('RAVENOUS_LANGUAGETOOL_STARTUP_TIMEOUT_SECONDS', '0.01')
    clients = mock_client(monkeypatch, lambda _request: httpx.Response(503))

    with pytest.raises(RuntimeError, match='did not become ready'):
        asyncio.run(grammar.start_checker())

    assert clients[0].is_closed
    assert grammar._client is None
    assert grammar._semaphore is None


def test_disabled_correction_never_connects(monkeypatch):
    monkeypatch.setenv('RAVENOUS_INPUT_CORRECTION_ENABLED', 'false')
    clients = mock_client(monkeypatch, lambda _request: pytest.fail('Unexpected request'))

    async def exercise():
        await grammar.start_checker()
        result = await grammar.correct_spelling_grammar('teh message')
        assert result.reason == 'disabled'
        await grammar.stop_checker()

    asyncio.run(exercise())
    assert clients == []


def test_outage_falls_back_and_recovers_with_same_client(monkeypatch):
    state = {'available': True}

    def handler(request):
        if not state['available']:
            raise httpx.ConnectError('service restarting', request=request)
        return httpx.Response(200, json={'matches': []})

    clients = mock_client(monkeypatch, handler)

    async def exercise():
        await grammar.start_checker()
        state['available'] = False
        failed = await grammar.correct_spelling_grammar('Original prompt')
        assert failed.reason == 'unavailable'
        assert failed.text == 'Original prompt'
        state['available'] = True
        recovered = await grammar.correct_spelling_grammar('Original prompt')
        assert recovered.status == 'unchanged'
        assert len(clients) == 1
        await grammar.stop_checker()

    asyncio.run(exercise())


def test_cancelling_startup_closes_client(monkeypatch):
    entered = asyncio.Event()

    async def handler(_request):
        entered.set()
        await asyncio.Event().wait()

    clients = mock_client(monkeypatch, handler)

    async def exercise():
        task = asyncio.create_task(grammar.start_checker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert clients[0].is_closed
    assert grammar._client is None


def test_slow_readiness_request_obeys_total_startup_deadline(monkeypatch):
    monkeypatch.setenv('RAVENOUS_LANGUAGETOOL_STARTUP_TIMEOUT_SECONDS', '0.01')

    async def handler(_request):
        await asyncio.Event().wait()

    clients = mock_client(monkeypatch, handler)

    async def exercise():
        with pytest.raises(RuntimeError, match='did not become ready'):
            await asyncio.wait_for(grammar.start_checker(), timeout=0.5)

    asyncio.run(exercise())
    assert clients[0].is_closed


@pytest.mark.parametrize('value', ['ftp://service', 'http://user:secret@host', 'http://'])
def test_invalid_endpoint_fails_without_echoing_credentials(monkeypatch, value):
    monkeypatch.setenv('RAVENOUS_LANGUAGETOOL_BASE_URL', value)
    with pytest.raises(RuntimeError, match='must be an HTTP') as failure:
        asyncio.run(grammar.start_checker())
    assert value not in str(failure.value)


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '3601', 'invalid'])
def test_startup_timeout_must_be_positive_and_finite(monkeypatch, value):
    monkeypatch.setenv('RAVENOUS_LANGUAGETOOL_STARTUP_TIMEOUT_SECONDS', value)
    with pytest.raises(RuntimeError, match='startup timeout'):
        asyncio.run(grammar.start_checker())
