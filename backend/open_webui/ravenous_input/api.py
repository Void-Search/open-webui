"""Authenticated Open WebUI API for repository prompt-preparation clients."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from open_webui.utils.auth import get_verified_user

from .pipeline import prepare_prompt

router = APIRouter()


class PreparePromptForm(BaseModel):
    text: str
    plain_text: bool = False


@router.post('/prepare')
async def prepare_prompt_endpoint(
    form_data: PreparePromptForm,
    _user=Depends(get_verified_user),
):
    """Prepare one prompt without returning its raw input."""
    result = await prepare_prompt(form_data.text, plain_text=form_data.plain_text)
    return result.public_dict()
