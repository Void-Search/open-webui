"""Clarification guidance for the native browser tool, before context fitting."""

CLARIFICATION_GUIDANCE = (
    'Use tools when they supply needed external information or perform a requested action. '
    'Otherwise complete the task directly in chat using the available context. '
    'When the request permits routine content choices or reversible defaults, choose reasonably and begin. '
    'Do not call ask_user for optional personalization, routine content choices, or to deliver requested '
    'interactive content. Ask only when missing information, permission, or a user-owned decision is '
    'necessary for a correct, safe next step, or the user explicitly requests that choice. '
    'For required clarification, use ask_user only with two or three genuine, distinct choices; '
    'otherwise ask a brief plain-text question. Do not invent preferences, earlier context, or choices. '
    'Follow the requested count, format and interaction. For interactive requests, provide the current '
    'requested step and wait for the user before evaluating their contribution or revealing information '
    'they were asked to work out.'
)


def clarification_guidance(tools, advertised, messages):
    """Return guidance only for an advertised native builtin, without duplicating it."""
    tool = tools.get('ask_user') or {}
    if tool.get('type') != 'builtin' or tool.get('tool_id') != 'builtin:ask_user':
        return None
    if not any(spec.get('function', {}).get('name') == 'ask_user' for spec in advertised):
        return None
    for message in messages:
        if message.get('role') != 'system':
            continue
        content = message.get('content')
        if isinstance(content, list):
            content = ' '.join(part['text'] for part in content if isinstance(part, dict)
                               and part.get('type') == 'text' and isinstance(part.get('text'), str))
        if isinstance(content, str) and CLARIFICATION_GUIDANCE in ' '.join(content.split()):
            return None
    return CLARIFICATION_GUIDANCE
