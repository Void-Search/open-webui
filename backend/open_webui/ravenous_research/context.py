"""Count the actual provider payload without issuing a generation request."""

import asyncio
import copy
import time
from collections import Counter
from contextvars import ContextVar

from ravenous_common.context import ContextBudgetError

from .evidence import context_message

probing = ContextVar('research_context_probe', default=False)


async def fit(request, body, user, ordered, question):
    from open_webui.utils.chat import generate_chat_completion
    from open_webui.utils.misc import merge_system_messages, strip_empty_content_blocks

    model = request.app.state.MODELS.get(body['model'], {})
    if getattr(request.state, 'direct', False) or model.get('owned_by') in ('ollama', 'arena') or model.get('pipe'):
        raise ContextBudgetError('Context counting is unavailable for this model')

    async def check(count):
        probe = {
            **copy.deepcopy({key: value for key, value in body.items() if key != 'metadata'}),
            'stream': False,
            'metadata': dict(body.get('metadata', {})),
            'messages': [
                *copy.deepcopy(body['messages']),
                context_message(
                    ordered[:count], question, body.get('metadata', {}).get('ravenous_evidence_assessment')
                ),
            ],
        }
        probe['messages'] = merge_system_messages(strip_empty_content_blocks(probe['messages']))
        token = probing.set(True)
        try:
            result = await generate_chat_completion(request, probe, user)
            if not isinstance(result, dict) or not result.get('research_context_checked'):
                raise ContextBudgetError('Context counting is unavailable for this model')
            return result['payload']['messages']
        finally:
            probing.reset(token)

    low, high, messages = 0, len(ordered), None
    # Usually the complete evidence set fits; only overflowing sets need binary search.
    try:
        messages = await check(high)
        low = high
    except ContextBudgetError as exc:
        if 'exceeds context' not in str(exc):
            raise
        high -= 1
    while low < high:
        middle = (low + high + 1) // 2
        try:
            messages = await check(middle)
            low = middle
        except ContextBudgetError as exc:
            if 'exceeds context' not in str(exc):
                raise
            high = middle - 1
    if messages is None:
        messages = await check(0)
    return ordered[:low], messages


def without_evidence(messages, original):
    result = []
    for message in messages:
        text = message.get('content')
        if message.get('role') == 'system' and isinstance(text, str) and original in text:
            text = text.replace(original, '').strip()
            if not text:
                continue
            message = {**message, 'content': text}
        result.append(message)
    return result


async def fit_by_deadline(request, body, user, selected, question, deadline):
    try:
        fitted, _ = await asyncio.wait_for(
            fit(request, body, user, selected, question), max(0.01, deadline - time.monotonic())
        )
        return fitted
    except (ContextBudgetError, TimeoutError):
        return []


async def finalize(request, body, user, emit):
    """Fit again after native tool definitions and system instructions are assembled."""
    from open_webui.utils.misc import merge_system_messages, strip_empty_content_blocks

    from .conversation import remember_result
    from .evidence import fallback_message, source_groups

    metadata = body['metadata']
    selected = metadata.get('ravenous_selected_passages', [])
    original = metadata.get('ravenous_evidence_message', '')
    if not selected or not original:
        return metadata.get('ravenous_retrieval_sources', [])
    question = metadata['ravenous_research_context']['query']
    messages = without_evidence(body['messages'], original)
    deadline = metadata.get('ravenous_research_deadline', time.monotonic() + 10)
    fitted = await fit_by_deadline(request, {**body, 'messages': messages}, user, selected, question, deadline)
    if len(fitted) != len(selected):
        assessment = dict(metadata.get('ravenous_evidence_assessment') or {})
        assessment.update(
            sufficient=False,
            missing=[*(assessment.get('missing') or []), 'Some supporting evidence exceeded the final context budget.'],
        )
        metadata['ravenous_evidence_assessment'] = assessment
        if fitted:
            # The partial-answer instruction also consumes tokens. Count exactly
            # the text that will be sent, including that new instruction.
            fitted = await fit_by_deadline(request, {**body, 'messages': messages}, user, fitted, question, deadline)
    message = context_message(fitted, question, metadata.get('ravenous_evidence_assessment'))
    body['messages'] = merge_system_messages(strip_empty_content_blocks([*messages, message]))
    metadata['ravenous_evidence_message'] = message['content']
    metadata['ravenous_selected_passages'] = fitted
    groups = source_groups(fitted)
    metadata['ravenous_retrieval_sources'] = groups
    if len(fitted) != len(selected):
        state = metadata['ravenous_research']
        report = state['report']
        numbers = {group['metadata'][0]['source']: index for index, group in enumerate(groups, 1)}
        counts = Counter(item['source_id'] for item in fitted)
        for source in report['sources']:
            source['citation'] = numbers.get(source['id'])
            source['selected_passages'] = counts[source['id']]
            if source['selected'] and source['id'] not in numbers:
                source['selected'] = False
                source['reasons'].append('context_limit')
        notice = 'Some supporting evidence was omitted after accounting for the full model context.'
        report['summary'] += ' ' + notice
        if not fitted:
            source_limited = metadata.get('ravenous_source_limited', False)
            report['answer_basis'] = 'source_limited' if source_limited else 'general_knowledge'
            if not source_limited:
                report['summary'] = (
                    'Answer basis: general knowledge; retrieval did not verify this answer. ' + report['summary']
                )
            fallback = fallback_message(question, report['summary'], source_limited=source_limited)
            body['messages'] = merge_system_messages(strip_empty_content_blocks([*messages, fallback]))
        recovery = state.get('recovery') or {
            'question': 'How would you like to continue?',
            'choices': [
                {
                    'label': 'Retry retrieval',
                    'reply': 'Retry web and local sources for the original question.',
                    'mode': 'both',
                }
            ],
        }
        result = {
            'pipeline': 'joint',
            'answer_basis': report.get('answer_basis', 'retrieved'),
            'query': question,
            'sufficient': False,
            'report': report,
            'recovery': recovery,
            'clarification_question': recovery['question'],
            'response_notice': report['summary'],
            'attempts': [{'query': value} for value in state.get('attempted_queries', [])],
        }
        await remember_result(metadata, user, result)
        if emit:
            await emit(
                {
                    'type': 'status',
                    'data': {'action': 'research_complete', 'report': report, 'recovery': recovery, 'done': True},
                }
            )
    return groups
