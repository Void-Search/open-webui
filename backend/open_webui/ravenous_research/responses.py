"""Completed research replies and clarification delivery for JSON and SSE clients."""

import codecs
import html
import json
import time
import uuid

from starlette.responses import StreamingResponse

from .conversation import append_question, finish_output


def excerpt_text(selected):
    """Quote final authorized passages without rewriting facts or source boundaries."""
    identifiers, excerpts = {}, []
    # Selection is already bounded by verification and the final context budget.
    # Keep complete passages so a limit here cannot detach a qualifying line.
    punctuation = str.maketrans({char: f'&#{ord(char)};' for char in '\\`*_{}[]()#+-.!|~'})
    for item in selected:
        number = identifiers.setdefault(item['source_id'], len(identifiers) + 1)
        text = item.get('text')
        if not isinstance(text, str) or not text.strip():
            continue
        # Entities preserve visible wording while disabling Markdown links,
        # images, headings and source-authored citation markers.
        escaped = html.escape(text, quote=False).translate(punctuation)
        quote = '\n'.join('> ' + line if line else '>' for line in escaped.splitlines())
        excerpts.append(quote + f'\n\n[{number}]')
    if not excerpts:
        return 'I could not verify enough source detail to answer this request.'
    return (
        'Here are relevant source excerpts. Details not stated in them remain unverified; this is not '
        'a complete list or a guarantee that every item is new.\n\n'
        + '\n\n'.join(excerpts)
    )


def excerpt_response(selected, model, stream=False):
    """Use the ordinary completion processor for source events, persistence and SSE completion."""
    text = excerpt_text(selected)
    envelope = {
        'id': 'chatcmpl-ravenous-' + uuid.uuid4().hex,
        'created': int(time.time()),
        'model': model,
    }
    if not stream:
        return {
            **envelope,
            'object': 'chat.completion',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        }

    async def events():
        for delta, finish in [({'role': 'assistant', 'content': text}, None), ({}, 'stop')]:
            yield 'data: ' + json.dumps({
                **envelope,
                'object': 'chat.completion.chunk',
                'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}],
            }) + '\n\n'
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
