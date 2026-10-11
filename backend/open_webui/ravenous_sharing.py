"""Keep service-owned personal context out of shared reply workflows."""

import os

from fastapi import HTTPException


def personal_workflows():
    return any(os.environ.get(name) == 'true' for name in (
        'RAVENOUS_MEMORY_ENABLED', 'RAVENOUS_ADAPTIVE_ENABLED',
    ))


def require_clone_only(share_mode):
    if personal_workflows() and share_mode == 'continue':
        raise HTTPException(400, 'Shared replies are unavailable while Ravenous personal workflows are enabled')


def require_chat_owner(chat, user):
    if personal_workflows() and chat.user_id != user.id:
        raise HTTPException(403, 'Clone this chat before replying or resolving its tools')
