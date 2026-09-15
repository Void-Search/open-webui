"""Conservative prompt normalization that preserves literal spans."""

from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

from .protected import protect_text

_BLOCK_TAGS = {
    'address',
    'article',
    'blockquote',
    'br',
    'div',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'li',
    'p',
    'pre',
    'section',
    'tr',
}


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append('\n')

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append('\n')

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f'&{name};')

    def handle_charref(self, name: str) -> None:
        self.parts.append(f'&#{name};')


def _plain_text(text: str) -> str:
    without_fences = re.sub(r'(?m)^[ \t]*(?:```|~~~)[^\r\n]*(?:\r?\n|$)', '', text)
    parser = _PlainTextParser()
    parser.feed(without_fences)
    parser.close()
    return html.unescape(''.join(parser.parts))


def _normalize_controls(text: str) -> str:
    normalized = []
    for character in unicodedata.normalize('NFKC', text):
        if character in {'\n', '\t'}:
            normalized.append(character)
        elif unicodedata.category(character).startswith('C'):
            normalized.append(' ')
        else:
            normalized.append(character)
    return ''.join(normalized)


def _normalize_line(line: str) -> str:
    leading_match = re.match(r'[ \t]*', line)
    leading = leading_match.group().replace('\t', '    ') if leading_match else ''
    body = line[len(leading_match.group()) :] if leading_match else line
    hard_break = len(body) - len(body.rstrip(' \t')) >= 2
    body = re.sub(r'[ \t]+', ' ', body.rstrip(' \t'))
    if not body:
        return ''
    return f'{leading}{body}{"  " if hard_break else ""}'


def clean_prompt(text: str, *, plain_text: bool = False) -> str:
    """Normalize prose without rewriting protected prompt literals."""
    source = _plain_text(text) if plain_text else text
    protected = protect_text(source)
    normalized = _normalize_controls(protected.masked).replace('\r\n', '\n').replace('\r', '\n')
    normalized = '\n'.join(_normalize_line(line) for line in normalized.split('\n'))
    restored = protected.restore(normalized)
    return source if restored is None else restored
