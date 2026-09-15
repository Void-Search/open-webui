"""Tests for conservative prompt cleanup and protected spans."""

from open_webui.ravenous_input.cleanup import clean_prompt
from open_webui.ravenous_input.protected import protect_text


def test_cleanup_normalizes_prose_without_flattening_markdown():
    source = 'Ｈｅｌｌｏ\t  world  \r\n\r\n-   item  \r\n'

    assert clean_prompt(source) == 'Hello world  \n\n- item  \n'


def test_cleanup_preserves_literals_and_protected_terms_exactly():
    source = 'Fix   this around Open WebUI and https://example.test/a_b?q=2 😀.\n\n```python\r\nvalue  =  2\r\n```'

    assert clean_prompt(source) == (
        'Fix this around Open WebUI and https://example.test/a_b?q=2 😀.\n\n```python\r\nvalue  =  2\r\n```'
    )


def test_cleanup_preserves_identifiers_names_quantities_and_negation():
    source = 'Do   not rename request_id, HTTP_API, Mary Jones, or 12.5kg.'

    assert clean_prompt(source) == 'Do not rename request_id, HTTP_API, Mary Jones, or 12.5kg.'


def test_plain_text_removes_tags_and_fences_but_decodes_entities_once():
    source = '<p>A &amp;amp; B</p>\n```text\nkept <span>literal</span>\n```'

    result = clean_prompt(source, plain_text=True)

    assert '<p>' not in result
    assert '```' not in result
    assert 'A &amp; B' in result
    assert 'kept' in result
    assert 'literal' in result


def test_restore_rejects_a_changed_or_duplicated_placeholder():
    protected = protect_text('Use `some_code` now')
    token = protected.spans[0].token

    assert protected.restore(protected.masked.replace(token, 'changed')) is None
    assert protected.restore(f'{protected.masked} {token}') is None
