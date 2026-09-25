"""Completed research replies and clarification delivery for JSON and SSE clients."""

import codecs
import json

from starlette.responses import StreamingResponse

from .conversation import append_question, finish_output


def complete_response(text, model, stream=False):
    result = {
        'model': model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
    }
    if not stream:
        return result

    async def events():
        yield (
            'data: '
            + json.dumps(
                {
                    'model': model,
                    'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}],
                }
            )
            + '\n\n'
        )
        yield (
            'data: '
            + json.dumps({'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
            + '\n\n'
        )
        yield 'data: [DONE]\n\n'

    return StreamingResponse(events(), media_type='text/event-stream')


def finish_json(data, metadata):
    state = metadata.get('ravenous_research') or {}
    if not (state.get('question') or state.get('response_notice')):
        return data
    for choice in data.get('choices', []):
        message = choice.get('message', {})
        if message.get('role') == 'assistant' and not message.get('tool_calls'):
            message['content'] = append_question(message.get('content') or '', metadata)
    if data.get('output'):
        data['output'] = finish_output(data['output'], metadata)
    return data


def _frame_processor(metadata):
    content = ''
    appended = False

    def process(frame):
        nonlocal content, appended
        if not frame.startswith('data:'):
            return frame + '\n\n'
        payload = frame[5:].strip()
        try:
            data = json.loads(payload)
        except ValueError:
            if payload != '[DONE]' or appended:
                return frame + '\n\n'
            updated = append_question(content, metadata)
            suffix = updated[len(content) :]
            appended = True
            addition = (
                (
                    'data: '
                    + json.dumps({'choices': [{'index': 0, 'delta': {'content': suffix}, 'finish_reason': None}]})
                    + '\n\n'
                )
                if suffix
                else ''
            )
            return addition + frame + '\n\n'
        choices = data.get('choices', []) if isinstance(data, dict) else []
        for choice in choices[:1]:
            delta = choice.get('delta', {})
            content += delta.get('content') or ''
            if choice.get('finish_reason') == 'stop' and not appended:
                updated = append_question(content, metadata)
                suffix = updated[len(content) :]
                if suffix:
                    delta['content'] = (delta.get('content') or '') + suffix
                    choice['delta'] = delta
                appended = True
        return 'data: ' + json.dumps(data) + '\n\n'

    return process


async def question_stream(original, metadata):
    """Inject before the terminal frame; tolerate arbitrarily split UTF-8/SSE chunks."""
    decoder = codecs.getincrementaldecoder('utf-8')()
    buffered = ''
    process = _frame_processor(metadata)
    try:
        async for chunk in original:
            buffered += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
            buffered = buffered.replace('\r\n', '\n')
            while '\n\n' in buffered:
                frame, buffered = buffered.split('\n\n', 1)
                yield process(frame)
        buffered += decoder.decode(b'', final=True)
        if buffered.strip():
            yield process(buffered)
    finally:
        if hasattr(original, 'aclose'):
            await original.aclose()
