"""Validated runtime settings rendered from Ravenous host configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class InputSettings:  # pylint: disable=too-many-instance-attributes
    correction_enabled: bool
    language: str
    correction_timeout_ms: int
    correction_max_characters: int
    correction_concurrency: int
    cleanup_profile: str
    default_output_tokens: int
    context_margin_tokens: int
    context_timeout_seconds: int


def _positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0 if allow_zero else value <= 0:
        raise RuntimeError(f'invalid {name}')
    return value


def load_settings() -> InputSettings:
    enabled = os.getenv('RAVENOUS_INPUT_CORRECTION_ENABLED', 'true').lower()
    if enabled not in {'true', 'false'}:
        raise RuntimeError('invalid RAVENOUS_INPUT_CORRECTION_ENABLED')
    cleanup_profile = os.getenv('RAVENOUS_INPUT_CLEANUP_PROFILE', 'conservative')
    if cleanup_profile != 'conservative':
        raise RuntimeError('invalid RAVENOUS_INPUT_CLEANUP_PROFILE')
    return InputSettings(
        correction_enabled=enabled == 'true',
        language=os.getenv('RAVENOUS_INPUT_LANGUAGE', 'en-AU'),
        correction_timeout_ms=_positive_int('RAVENOUS_INPUT_CORRECTION_TIMEOUT_MS', 200),
        correction_max_characters=_positive_int('RAVENOUS_INPUT_CORRECTION_MAX_CHARACTERS', 2000),
        correction_concurrency=_positive_int('RAVENOUS_INPUT_CORRECTION_CONCURRENCY', 2),
        cleanup_profile=cleanup_profile,
        default_output_tokens=_positive_int('RAVENOUS_INPUT_DEFAULT_OUTPUT_TOKENS', 2048),
        context_margin_tokens=_positive_int('RAVENOUS_INPUT_CONTEXT_MARGIN_TOKENS', 64, allow_zero=True),
        context_timeout_seconds=_positive_int('RAVENOUS_INPUT_CONTEXT_TIMEOUT_SECONDS', 10),
    )
