"""Prepare saved local chats and publish only completed, owned transcripts."""

import copy
import hashlib
import logging
from collections import Counter

import httpx
from ravenous_common.context import is_local_provider, measure_context
from ravenous_common.memory import apply_context, receipt
from ravenous_common.settings import load_settings

from .client import client, configured

log = logging.getLogger(__name__)


def remember_base_prompt(metadata):
    """Restore recalled context without adding research to the source-free prompt."""
    context = metadata.get('ravenous_context') or {}
    recalled = apply_context([], context)
    if recalled:
        base = metadata.get('base_system_prompt') or ''
        metadata['base_system_prompt'] = '\n\n'.join(
            part for part in (base, recalled[0]['content']) if part
        )


def branch(messages, mapping):
    forks = []
    siblings = Counter(item.get('parentId') for item in mapping.values())
    for row in messages:
        # Provider replay omits parentId; resolve it from the owned transcript.
        parent = mapping.get(row.get('id'), row).get('parentId')
        if siblings[parent] > 1:
            forks.append(row.get('id', ''))
    return hashlib.sha256('|'.join(forks).encode()).hexdigest() if forks else 'main'


def snapshot(messages):
    from open_webui.utils.misc import get_output_text

    return [{**row, 'content': get_output_text(row['output']) if row.get('output') else row.get('content', '')} for row in messages]


async def connection(request, model):
    if not configured() or getattr(request.state, 'direct', False) or model.get('owned_by') != 'openai':
        return None
    from open_webui.routers.openai import get_openai_connection

    model_id = (model.get('info') or {}).get('base_model_id') or model['id']
    resolved = request.app.state.OPENAI_MODELS.get(model_id, model)
    if not isinstance(resolved.get('urlIdx'), int):
        return None
    base, key, config = await get_openai_connection(resolved['urlIdx'])
    if not is_local_provider(base):
        return None
    prefix = config.get('prefix_id')
    if prefix and model_id.startswith(prefix + '.'):
        model_id = model_id[len(prefix) + 1:]
    return base, key, model_id


async def allowed_memory(user, requested):
    if not requested:
        return False
    from open_webui.models.config import Config
    from open_webui.utils.access_control import has_permission

    return bool(await Config.get('memories.enable')) and (
        user.role == 'admin' or await has_permission(user.id, 'features.memories', await Config.get('user.permissions'))
    )


async def own_chat(user, metadata):
    from open_webui.models.chats import Chats
    from open_webui.utils.chat_id import is_saved_chat_id

    chat_id = metadata.get('chat_id')
    if not is_saved_chat_id(chat_id) or not await Chats.get_chat_by_id_and_user_id(chat_id, user.id):
        metadata['ravenous_memory_excluded'] = True
        return False
    metadata['ravenous_memory_owned'] = True
    return True


async def prepare(request, user, form_data, metadata, target, *, model=None, originals=None, requested_memory=None):
    from open_webui.models.chats import Chats

    if not await own_chat(user, metadata):
        return form_data['messages']
    originals = snapshot(copy.deepcopy(originals if originals is not None else form_data['messages']))
    capabilities = ((model or {}).get('info', {}).get('meta', {}).get('capabilities') or {})
    if requested_memory is None:
        requested_memory = (form_data.get('features') or {}).get('memory')
    metadata['ravenous_memory_enabled'] = capabilities.get('memory', True) and await allowed_memory(user, requested_memory)
    chat_id = metadata['chat_id']
    mapping = await Chats.get_messages_map_by_chat_id(chat_id)
    metadata['ravenous_memory_branch'] = branch(originals, mapping or {})
    options = {'session_id': chat_id, 'branch_id': metadata['ravenous_memory_branch'], 'memory_enabled': metadata['ravenous_memory_enabled']}
    try:
        base, key, model_id = target
        payload = {key: form_data[key] for key in ('tools', 'tool_choice', 'max_tokens', 'chat_template_kwargs') if key in form_data}
        payload.update({'model': model_id, 'messages': [{key: value for key, value in row.items() if key in ('role', 'content', 'tool_calls', 'tool_call_id')} for row in form_data['messages']]})
        async with httpx.AsyncClient(timeout=10, trust_env=False) as inference:
            measured = await measure_context(payload, base_url=base, api_key=key, client=inference, settings=load_settings())
        result = await client(user).prepare(options, originals, measured)
        metadata['ravenous_context'] = receipt(result)
        return apply_context(form_data['messages'], result)
    except Exception:
        # Provider budgeting still checks the exact final prompt. Failure cannot
        # activate a second compaction/memory owner or publish partial history.
        metadata['ravenous_context'] = {'summary': '', 'memories': [], 'notices': ['context_unavailable']}
        log.info('Magic Context unavailable; using original chat history')
        return form_data['messages']


async def completed(request, user, metadata, messages):
    if not metadata.get('ravenous_memory_owned') or not metadata.get('ravenous_memory_enabled'):
        return
    from open_webui.models.chats import Chats

    if not await Chats.get_chat_by_id_and_user_id(metadata['chat_id'], user.id):
        return
    if not messages or messages[-1].get('role') != 'assistant' or not messages[-1].get('done') or messages[-1].get('error'):
        return
    try:
        await client(user).completed({'session_id': metadata['chat_id'], 'branch_id': metadata.get('ravenous_memory_branch', 'main'), 'memory_enabled': True}, snapshot(messages))
    except Exception:
        log.info('Magic Context capture unavailable; original transcript retained')


async def publish(request, user, metadata, event_emitter=None):
    context = metadata.get('ravenous_context')
    if context is None:
        return
    from open_webui.models.chats import Chats

    if metadata.get('ravenous_memory_owned') and await Chats.get_chat_by_id_and_user_id(metadata['chat_id'], user.id):
        previous = await Chats.get_message_by_id_and_message_id(metadata['chat_id'], metadata['message_id']) or {}
        await Chats.upsert_message_to_chat_by_id_and_message_id(metadata['chat_id'], metadata['message_id'], {'meta': {**(previous.get('meta') or {}), 'ravenous_context': context}}, touch=False)
    if event_emitter:
        await event_emitter({'type': 'chat:completion', 'data': {'ravenous_context': context}})
