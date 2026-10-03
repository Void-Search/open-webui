"""Completed research replies and clarification delivery for JSON and SSE clients."""

import codecs
import json
import re
import time
import uuid

from starlette.responses import StreamingResponse

from .conversation import append_question, finish_output


def excerpt_text(selected, limited=False):
    """Quote final authorized passages without rewriting facts or source boundaries."""
    identifiers, excerpts = {}, []
    # Selection is already bounded by verification and the final context budget.
    # Keep complete passages so a limit here cannot detach a qualifying line.
    links = re.compile(r'!?\[([^\]\n]*)\]\((?:\\.|[^\\()\n]|\([^()\n]*\))*\)')
    for item in selected:
        number = identifiers.setdefault(item['source_id'], len(identifiers) + 1)
        text = item.get('text')
        if not isinstance(text, str) or not text.strip():
            continue
        # A literal fence protects source punctuation from WebUI's math and
        # citation extensions, including indented source lines. Source backticks
        # cannot close a fence longer than any run present in the passage.
        plain = links.sub(lambda match: match[1], text)
        fence = '`' * max(3, max((len(run) for run in re.findall(r'`+', plain)), default=0) + 1)
        literal = f'{fence}ravenous-excerpt\n{plain}\n{fence}'
        quote = '\n'.join('> ' + line if line else '>' for line in literal.splitlines())
        excerpts.append(quote + f'\n\n[{number}]')
    if not excerpts:
        return 'I could not verify enough source detail to answer this request.'
    introduction = (
        'These sources only partly support the request. Details from the earlier answer that are not '
        'supported below remain unverified.'
        if limited
        else 'Here are relevant source excerpts. Details not stated in them remain unverified.'
    )
    return (
        introduction + ' This is not a complete list or a guarantee that every item is new.\n\n'
        + '\n\n'.join(excerpts)
    )


def excerpt_response(selected, model, stream=False, limited=False):
    """Use the ordinary completion processor for source events, persistence and SSE completion."""
    text = excerpt_text(selected, limited=limited)
    return completion_response(text, model, stream)


def details_text(details, selected, limited=False):
    """Present literal lookup values without dumping unrelated source sections.

    Complete sections remain attached to citations. A compact value cannot omit
    a recognizable governing condition; keep the complete quotation in that case.
    """
    identifiers = {}
    for item in selected:
        identifiers.setdefault(item['source_id'], len(identifiers) + 1)
    values = {}
    for detail in details:
        entity, value, source_id = detail['entity'], detail['value'], detail['source_id']
        if source_id not in identifiers:
            continue
        records = [item['text'] for item in selected if item['source_id'] == source_id
                   and ' '.join(value.casefold().split()) in ' '.join(item['text'].casefold().split())]
        if not records:
            continue
        for record in records:
            for paragraph in re.split(r'\n\s*\n', record):
                if re.search(r'\b(?:only|unless|except|requires?|required|must|without)\b|subject to|provided that',
                             paragraph, re.I) and ' '.join(paragraph.split()) not in ' '.join(value.split()):
                    return None
        # Keep the most complete copied value for each requested name.
        if entity not in values or len(value) > len(values[entity]['value']):
            values[entity] = detail
    if not values:
        return None
    sections = []
    for detail in values.values():
        entity = re.sub(r'([\\`*_\[\]])', r'\\\1', detail['entity'])
        text = '**' + entity + '**\n\n' + detail['value']
        fence = '`' * max(3, max((len(run) for run in re.findall(r'`+', text)), default=0) + 1)
        sections.append(f'{fence}ravenous-excerpt\n{text}\n{fence}\n\n[{identifiers[detail["source_id"]]}]')
    return ('These are the requested details stated in the sources:\n\n' + '\n\n'.join(sections)
            + ('\n\nI could not verify the remaining requested details.' if limited else ''))


def completion_response(text, model, stream=False, usage=None):
    """Deliver a completed answer after validation, in the client's original format."""
    envelope = {
        'id': 'chatcmpl-ravenous-' + uuid.uuid4().hex,
        'created': int(time.time()),
        'model': model,
    }
    if not stream:
        return {
            **envelope,
            **({'usage': usage} if usage else {}),
            'object': 'chat.completion',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        }

    async def events():
        for delta, finish in [({'role': 'assistant', 'content': text}, None), ({}, 'stop')]:
            yield 'data: ' + json.dumps({
                **envelope,
                **({'usage': usage} if usage and finish else {}),
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
