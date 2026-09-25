"""Request constraints and truthful execution guidance for native research chat."""

import re

from fastapi import HTTPException


class ResearchUnavailable(HTTPException):
    """A controlled search outcome rendered as a completed assistant message."""

    def __init__(self, result=None):
        result = result or unavailable_result()
        super().__init__(503, recovery_message(result))


def unavailable_result():
    return {
        'sufficient': False,
        'evidence': [],
        'supporting_passages': [],
        'attempts': [],
        'stop_reason': 'provider_unavailable',
        'clarification_question': 'Would you like me to try again, or can you paste some source text to work from?',
    }


def failure_codes(result):
    return {code for attempt in result.get('attempts', []) for code in attempt.get('failure_codes', [])} | {
        result.get('stop_reason', '')
    }


def clarification_question(result):
    if result.get('pipeline') == 'joint':
        return result.get('clarification_question')
    if result.get('sufficient'):
        return None
    if failure_codes(result) & {'provider_unavailable', 'discovery_unavailable', 'fetch_unavailable'}:
        return unavailable_result()['clarification_question']
    question = (
        result.get('clarification_question')
        or 'Which detail should I focus on, or is there a source you would like me to check?'
    )
    # Keep the recovery an ordinary, single question even with a verbose small model.
    question = ' '.join(question.split()).split('?', 1)[0].strip()[:499]
    return question + '?'


def recovery_notice(result):
    if result.get('sufficient'):
        return 'Found sources for your question.'
    if result.get('supporting_passages'):
        return 'Found some relevant information; one detail needs clarification.'
    codes = failure_codes(result)
    if codes & {'provider_unavailable', 'discovery_unavailable', 'fetch_unavailable'}:
        return 'The web research service is temporarily unavailable.'
    if codes & {'deadline', 'discovery_timeout', 'fetch_timeout'}:
        return 'The search ran out of time before I could read enough information.'
    if any(code.startswith('fetch_') or code == 'response_too_large' for code in codes):
        return "I found pages, but couldn't read enough information from them."
    return "I haven't found enough information to answer that yet."


def recovery_message(result):
    if result.get('pipeline') == 'joint':
        summary = result.get('report', {}).get('summary', 'Retrieval could not be completed.')
        return 'I could not verify enough supporting evidence to answer.\n\n' + summary
    return recovery_notice(result) + '\n\n' + (clarification_question(result) or '')


_ALIASES = {
    'stack overflow': 'stackoverflow.com',
    'stackoverflow': 'stackoverflow.com',
    'youtube': 'youtube.com',
    'wikipedia': 'wikipedia.org',
    'reddit': 'reddit.com',
}


def requested_domains(question):
    # Only inspect the current instruction, not domains mentioned in older turns.
    current = question.split('\nEarlier user context', 1)[0].lower()
    if re.search(r"\b(?:don't|do not|never)\s+(?:search|check|use)\b", current):
        return []
    domains = re.findall(
        r'\b(?:site:|https?://|(?:on|from|search|check|use)\s+)((?:[a-z0-9-]+\.)+[a-z]{2,63})\b', current
    )
    for name, domain in _ALIASES.items():
        if re.search(r'\b(?:search|check|use|on|from)\s+(?:on\s+|the\s+)?' + re.escape(name) + r'\b', current):
            domains.append(domain)
    return list(dict.fromkeys(d.removeprefix('www.') for d in domains))[:10]


def initial_query(queries, question):
    context = question.split('\nEarlier user context (resolve references only):\n', 1)
    topic = context[-1] if len(context) > 1 else question
    terms = set(re.findall(r'\w{3,}', topic.lower())) - {'search', 'check', 'online', 'current', 'request'}
    candidates = [query for query in queries if isinstance(query, str) and query.strip()]
    return max(candidates, key=lambda q: len(terms & set(re.findall(r'\w{3,}', q.lower()))), default=topic)[:3500]


def execution_requested(messages):
    requests = [
        m.get('content', '') for m in messages if m.get('role') == 'user' and isinstance(m.get('content'), str)
    ][-3:]
    if requests and re.search(r"\b(?:don't|do not|no need to)\s+(?:run|execute|benchmark)", requests[-1], re.I):
        return False
    if not requests:
        return False
    followup = len(requests[-1].split()) <= 16 and re.match(
        r'(?:search|check|try|compare|which|what about)\b', requests[-1], re.I
    )
    active = requests if followup else requests[-1:]
    return any(
        re.search(
            r'\b(?:run|execute)\s+(?:the\s+|this\s+)?(?:python\s+)?code\b|\bbenchmark\b|\bmeasure\s+(?:performance|timings?)\b',
            text,
            re.I,
        )
        for text in active
    )


EXECUTION_GUIDANCE = """The user requested execution or a benchmark. Use execute_code before
claiming the examples work or naming a measured winner. Use the SAME fixture and
output type for every method. Assert each method equals an independently computed
expected result, including order, before timing it. Repair exceptions and rerun;
never present expected output comments as observed output. For comparisons run
repeated timings on fresh copies of identical inputs; print method names, fixture
size, correctness checks and measured durations. Report Python/runtime and that
browser timings are local, not universal. Distinguish methods from underlying
algorithms. If execution fails, is denied or unavailable, clearly state that no
verified benchmark was completed and do not invent timings or a fastest method.
Do not substitute toy demonstration code for the requested comparison."""


def execution_response(stdout, stderr, result):
    return {
        'status': 'error' if stderr else 'success',
        'stdout': stdout,
        'stderr': stderr,
        'result': result,
        'verification': (
            'Execution reported an error. Repair and rerun; no successful comparison is established.'
            if stderr
            else 'Execution completed. Only the printed checks and measurements are observed results.'
        ),
    }
