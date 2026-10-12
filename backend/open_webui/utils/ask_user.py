from collections.abc import Callable

from open_webui.utils.json_codec import JSONCodec

ASK_USER_NAME = 'ask_user'
ASK_USER_MAX_QUESTIONS = 3
ASK_USER_MIN_OPTIONS = 2
ASK_USER_MAX_OPTIONS = 3
ASK_USER_ID_MAX_LENGTH = 64
ASK_USER_HEADER_MAX_LENGTH = 48
ASK_USER_QUESTION_MAX_LENGTH = 500
ASK_USER_LABEL_MAX_LENGTH = 80
ASK_USER_DESCRIPTION_MAX_LENGTH = 240
ASK_USER_DEFAULT_TIMEOUT_MS = 120_000
ASK_USER_MIN_TIMEOUT_MS = 60_000
ASK_USER_MAX_TIMEOUT_MS = 240_000


def get_ask_user_parameters_schema() -> dict:
    """Describe the browser input contract to the model, including nested fields."""
    return {
        'type': 'object',
        'properties': {
            'questions': {
                'type': 'array',
                'description': 'Clarifying questions with distinct ids. Put the recommended option first.',
                'minItems': 1,
                'maxItems': ASK_USER_MAX_QUESTIONS,
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {
                            'type': 'string',
                            'description': 'A stable, distinct id used to identify the answer.',
                            'minLength': 1,
                            'maxLength': ASK_USER_ID_MAX_LENGTH,
                        },
                        'header': {
                            'type': 'string',
                            'description': 'Optional short heading displayed above the question.',
                            'maxLength': ASK_USER_HEADER_MAX_LENGTH,
                        },
                        'question': {
                            'type': 'string',
                            'description': 'The question the user needs to answer.',
                            'minLength': 1,
                            'maxLength': ASK_USER_QUESTION_MAX_LENGTH,
                        },
                        'options': {
                            'type': 'array',
                            'description': 'Two or three meaningful choices; put the recommended choice first.',
                            'minItems': ASK_USER_MIN_OPTIONS,
                            'maxItems': ASK_USER_MAX_OPTIONS,
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'label': {
                                        'type': 'string',
                                        'description': 'A short label for this choice.',
                                        'minLength': 1,
                                        'maxLength': ASK_USER_LABEL_MAX_LENGTH,
                                    },
                                    'description': {
                                        'type': 'string',
                                        'description': 'Explain what selecting this choice means.',
                                        'minLength': 1,
                                        'maxLength': ASK_USER_DESCRIPTION_MAX_LENGTH,
                                    },
                                },
                                'required': ['label', 'description'],
                                'additionalProperties': False,
                            },
                        },
                        'allow_other': {
                            'type': 'boolean',
                            'description': 'Override whether this question accepts a free-form answer.',
                        },
                    },
                    'required': ['id', 'question', 'options'],
                    'additionalProperties': False,
                },
            },
            'allow_other': {
                'type': 'boolean',
                'description': 'Allow a free-form answer in addition to the provided choices.',
                'default': True,
            },
            'timeout_ms': {
                'type': 'integer',
                'description': 'How long the browser keeps the prompt open, in milliseconds.',
                'minimum': ASK_USER_MIN_TIMEOUT_MS,
                'maximum': ASK_USER_MAX_TIMEOUT_MS,
                'default': ASK_USER_DEFAULT_TIMEOUT_MS,
            },
        },
        'required': ['questions'],
        'additionalProperties': False,
    }


def get_ask_user_tool_calls(tool_calls: list[dict]) -> tuple[list[dict], str | None]:
    ask_user_calls = [
        tool_call for tool_call in tool_calls if tool_call.get('function', {}).get('name') == ASK_USER_NAME
    ]
    if not ask_user_calls:
        return [], None
    if len(tool_calls) != 1:
        return (
            ask_user_calls,
            'Error: ask_user must be the only tool call, so it did not run. Call ask_user on its own.',
        )
    if len(ask_user_calls) != 1:
        return ask_user_calls, 'Error: only one ask_user call is allowed per turn.'
    return ask_user_calls, None


def normalize_ask_user_request(arguments: dict) -> dict:
    questions = arguments.get('questions')
    if not isinstance(questions, list) or not 1 <= len(questions) <= ASK_USER_MAX_QUESTIONS:
        raise ValueError('ask_user requires 1-3 questions.')

    normalized_questions = []
    seen_ids = set()
    allow_other = bool(arguments.get('allow_other', True))
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError('Each question must be an object.')

        question_id = str(question.get('id') or '').strip()[:ASK_USER_ID_MAX_LENGTH]
        if not question_id:
            raise ValueError('Each question requires a non-empty id.')
        if question_id in seen_ids:
            raise ValueError(f'Duplicate question id: {question_id}')
        seen_ids.add(question_id)

        options = question.get('options')
        if not isinstance(options, list) or not ASK_USER_MIN_OPTIONS <= len(options) <= ASK_USER_MAX_OPTIONS:
            raise ValueError('Each question requires 2-3 options.')

        normalized_options = []
        for option in options:
            if not isinstance(option, dict):
                raise ValueError('Each option must be an object.')
            label = str(option.get('label') or '').strip()[:ASK_USER_LABEL_MAX_LENGTH]
            description = str(option.get('description') or '').strip()[:ASK_USER_DESCRIPTION_MAX_LENGTH]
            if not label or not description:
                raise ValueError('Each option requires a label and description.')
            normalized_options.append({'label': label, 'description': description})

        question_text = str(question.get('question') or '').strip()[:ASK_USER_QUESTION_MAX_LENGTH]
        if not question_text:
            raise ValueError('Each question requires question text.')

        normalized_questions.append(
            {
                'id': question_id,
                'header': str(question.get('header') or '').strip()[:ASK_USER_HEADER_MAX_LENGTH]
                or f'Question {index + 1}',
                'question': question_text,
                'options': normalized_options,
                'allow_other': bool(question.get('allow_other', allow_other)),
            }
        )

    timeout_ms = arguments.get('timeout_ms', ASK_USER_DEFAULT_TIMEOUT_MS)
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or not ASK_USER_MIN_TIMEOUT_MS <= timeout_ms <= ASK_USER_MAX_TIMEOUT_MS
    ):
        timeout_ms = ASK_USER_DEFAULT_TIMEOUT_MS

    return {
        'questions': normalized_questions,
        'allow_other': allow_other,
        'timeout_ms': timeout_ms,
    }


def stage_ask_user_tool_calls(
    tool_calls: list[dict],
    output: list[dict],
    make_output_id: Callable[[str], str],
) -> tuple[bool, str | None]:
    ask_user_calls, error = get_ask_user_tool_calls(tool_calls)
    if not ask_user_calls:
        return False, None

    for tool_call in ask_user_calls:
        call_id = tool_call.get('id') or make_output_id('fc')
        raw_arguments = tool_call.get('function', {}).get('arguments', '{}')
        arguments = raw_arguments

        if not error:
            try:
                parsed_arguments = JSONCodec.loads(raw_arguments or '{}')
                if not isinstance(parsed_arguments, dict):
                    raise ValueError('ask_user arguments must be an object.')
                arguments = JSONCodec.dumps(normalize_ask_user_request(parsed_arguments))
            except (JSONCodec.JSONDecodeError, TypeError, ValueError) as exc:
                error = f'Error: {exc}'

        item = {
            'type': 'function_call',
            'id': call_id or make_output_id('fc'),
            'call_id': call_id,
            'name': ASK_USER_NAME,
            'arguments': arguments,
            'status': 'completed' if error else 'pending',
        }

        existing_item = next(
            (
                existing
                for existing in output
                if existing.get('type') == 'function_call'
                and (
                    existing.get('call_id') == call_id
                    or existing.get('id') == tool_call.get('id')
                    or (
                        not existing.get('call_id')
                        and existing.get('name') == ASK_USER_NAME
                        and existing.get('status') not in {'rejected', 'failed'}
                    )
                )
            ),
            None,
        )
        if existing_item:
            existing_item.update(item)
        else:
            output.append(item)

        # Every invalid call needs its own result, or the UI waits on it forever.
        if error:
            output.append(
                {
                    'type': 'function_call_output',
                    'id': make_output_id('fco'),
                    'call_id': call_id,
                    'output': [{'type': 'input_text', 'text': error}],
                    'status': 'completed',
                }
            )

    return True, error
