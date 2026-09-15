"""Mask prompt spans that correction and cleanup must preserve verbatim."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_GLOSSARY = Path(__file__).with_name('protected-terms.txt')
_PATTERNS = (
    re.compile(r'(?ms)^[ \t]*(?P<fence>```|~~~)[^\r\n]*(?:\r?\n).*?^[ \t]*(?P=fence)[ \t]*$'),
    re.compile(r'`[^`\r\n]+`'),
    re.compile(r'(?m)^(?: {4}|\t)[^\r\n]*(?:\r?\n|$)'),
    re.compile(r'</?[A-Za-z][^>]*>'),
    re.compile(r'(?<=\]\()[^\s)]+(?=\))'),
    re.compile(r'(?:https?://|www\.)[^\s<>()]+', re.IGNORECASE),
    re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.IGNORECASE),
    re.compile(r'(?<!\w)@[A-Za-z0-9_.-]+'),
    re.compile(r'(?<!\w)/[A-Za-z][A-Za-z0-9_-]*'),
    re.compile(r'\b(?:[A-Za-z0-9]+_[A-Za-z0-9_]+|[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*)\b'),
    re.compile(r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b'),
    re.compile(r'\b[A-Z][A-Z0-9_]{1,}\b'),
    re.compile(
        r'(?<!\w)[+-]?(?:\d+(?:[.,]\d+)*)(?:\s?(?:%|°[CF]|[kmgt]?i?[bB]|'
        r'ms|s|min|h|d|mm|cm|m|km|mg|g|kg|ml|l))?(?!\w)',
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:not|no|never|cannot|without|neither|nor)\b|\b\w+n['’]t\b", re.IGNORECASE),
    re.compile(
        '['
        '\U0001f1e6-\U0001f1ff'
        '\U0001f300-\U0001faff'
        '\u2600-\u27bf'
        '](?:[\ufe0e\ufe0f\u200d]|[\U0001f3fb-\U0001f3ff]|'
        '[\U0001f300-\U0001faff])*'
    ),
    re.compile(r'\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+\b'),
)


@dataclass(frozen=True)
class ProtectedSpan:
    """One source literal and its location in the masked prompt."""

    token: str
    value: str
    masked_start: int
    masked_end: int


@dataclass(frozen=True)
class ProtectedText:
    """A masked prompt with strict, one-shot literal restoration."""

    source: str
    masked: str
    spans: tuple[ProtectedSpan, ...]

    def overlaps(self, start: int, end: int) -> bool:
        for span in self.spans:
            if start == end:
                if span.masked_start < start < span.masked_end:
                    return True
            elif start < span.masked_end and end > span.masked_start:
                return True
        return False

    def restore(self, candidate: str) -> str | None:
        positions = []
        for span in self.spans:
            if candidate.count(span.token) != 1:
                return None
            positions.append(candidate.index(span.token))
        if positions != sorted(positions):
            return None
        restored = candidate
        for span in self.spans:
            restored = restored.replace(span.token, span.value, 1)
        return restored


def _glossary_patterns() -> tuple[re.Pattern[str], ...]:
    terms = []
    for line in _GLOSSARY.read_text(encoding='utf-8').splitlines():
        term = line.strip()
        if term and not term.startswith('#'):
            terms.append(re.compile(rf'(?<!\w){re.escape(term)}(?!\w)', re.IGNORECASE))
    return tuple(terms)


def _ranges(text: str) -> list[tuple[int, int]]:
    ranges = [match.span() for pattern in (*_PATTERNS, *_glossary_patterns()) for match in pattern.finditer(text)]
    if not ranges:
        return []
    merged = []
    for start, end in sorted(ranges, key=lambda item: (item[0], -item[1])):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def protect_text(text: str) -> ProtectedText:
    """Replace sensitive spans with deterministic tokens and retain exact originals."""
    prefix = '__RAVENOUS_LITERAL_'
    while prefix in text:
        prefix = f'_{prefix}'
    parts = []
    spans = []
    source_cursor = 0
    masked_length = 0
    for index, (start, end) in enumerate(_ranges(text)):
        prose = text[source_cursor:start]
        token = f'{prefix}{index:04d}__'
        parts.extend((prose, token))
        masked_start = masked_length + len(prose)
        spans.append(
            ProtectedSpan(
                token=token,
                value=text[start:end],
                masked_start=masked_start,
                masked_end=masked_start + len(token),
            )
        )
        masked_length = masked_start + len(token)
        source_cursor = end
    parts.append(text[source_cursor:])
    return ProtectedText(source=text, masked=''.join(parts), spans=tuple(spans))
