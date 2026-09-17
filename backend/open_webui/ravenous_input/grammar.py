"""Open WebUI lifecycle and correction exports from the shared package."""

from ravenous_common.grammar import (
    CorrectionResult,
    correct_spelling_grammar,
    start_checker,
    stop_checker,
    utf16_offset_to_index,
)

__all__ = [
    'CorrectionResult',
    'correct_spelling_grammar',
    'start_checker',
    'stop_checker',
    'utf16_offset_to_index',
]
