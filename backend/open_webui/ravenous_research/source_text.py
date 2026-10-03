"""Conservative removal of recognizable web-page furniture before verification."""

import re

LINK = re.compile(r'!?\[([^\]\n]*)\]\((?:\\.|[^\\()\n]|\([^()\n]*\))*\)')
UI_LABELS = {
    'advertisement', 'advertisements', 'advertising', 'skip to content',
    'skip to main content', 'menu', 'sign in', 'sign up', 'subscribe',
    'share', 'share this article', 'print', 'print this article',
}
FOOTER_LABELS = {
    'related articles', 'related posts', 'recommended articles', 'read next',
    'you may also like', 'more articles', 'about the author', 'free printable',
    'newsletter', 'subscribe to our newsletter',
}


def label(text):
    return LINK.sub(r'\1', text).strip(' \t#*_').casefold()


def promotional_paragraph(text):
    """Standalone offers, not advice that merely mentions a paid resource."""
    plain = label(text)
    offering = re.search(r'\b(?:course|newsletter|printable)\b', plain)
    action = re.search(r'\b(?:buy|purchase|enrol|enroll|join|subscribe|sign up|send me|waiting for you)\b', plain)
    conditional_offer = re.match(r'if you[’\']?re looking\b', plain) and LINK.search(text)
    return bool(offering and (conditional_offer or (action and re.search(r'\b(?:i|my|me|our|us)\b', plain))))


def trim_footer_labels(lines):
    """Remove adjoining display labels before an explicit footer/offer marker."""
    while lines:
        while lines and not lines[-1].strip():
            lines.pop()
        start = len(lines) - 1
        while start > 0 and lines[start - 1].strip():
            start -= 1
        block = '\n'.join(lines[start:])
        # Code, headings, lists, punctuation and qualification words remain.
        if (len(block) > 120 or re.search(r'[.!?:;]|^\s*(?:#{1,6}\s|`{3,}|~{3,}|\||[-*+]\s|\d+[.)]\s)|\b(?:only|require\w*|unless|except|must|without|not|if)\b', block, re.I | re.M)):
            break
        del lines[start:]


def clean_web_text(text):
    """Keep prose, conditions, lists and code; never rewrite their factual text.

    Only explicit UI labels, breadcrumb paths, standalone promotions and labelled
    footer sections are removed. Other ambiguous text remains for verification.
    """
    kept, fence, footer_level = [], None, None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        marker = re.match(r'^\s*(`{3,}|~{3,})(.*)$', line)
        if fence:
            kept.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker and footer_level is None:
            fence = marker[1]
            kept.append(line)
            continue
        plain = label(line)
        heading = re.match(r'^\s*(#{1,6})\s+', line)
        if footer_level is not None:
            if not heading or (footer_level and len(heading[1]) > footer_level):
                continue
            footer_level = None
        if plain in FOOTER_LABELS:
            if plain in {'free printable', 'newsletter'}:
                trim_footer_labels(kept)
            footer_level = len(heading[1]) if heading else 0
            continue
        if plain in UI_LABELS or re.fullmatch(r'(?:send me (?:the )?\w+|unsubscribe whenever[^.!]*[.!]?|[^.!?]{1,70} starts here)', plain):
            continue
        # Breadcrumb entries are short labels prefixed by a navigation separator.
        if re.fullmatch(r'[/›»]\s+[^.!?\n]{1,100}', plain) and len(plain.split()) <= 8:
            continue
        if plain == 'home' and any(
            re.match(r'\s*[/›»]\s+', following)
            for following in lines[index + 1:index + 5] if following.strip()
        ):
            continue
        kept.append(line)
    paragraphs = re.split(r'\n\s*\n', '\n'.join(kept))
    return '\n\n'.join(paragraph for paragraph in paragraphs if not promotional_paragraph(paragraph)).strip()


def article_body_record(text, source_url=''):
    """A link-leading teaser/card is not a passage from the article it advertises."""
    first = text.lstrip().splitlines()[0] if text.strip() else ''
    if first == source_url or re.match(r'\[#{1,6}\s', first):
        return True
    if re.search(r'^\s*(?:#{1,6}\s|`{3,}|~{3,}|[-*+]\s|\|)', text, re.M):
        return True
    if re.fullmatch(r'https?://\S+|\[[^\]]+\]\(https?://[^\n]+\)', first):
        return False
    return True
