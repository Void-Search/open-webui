"""Conservative excerpts preserve exact bounded source context and its qualifiers."""

import pytest
from open_webui.ravenous_research import evidence


def quote(text, anchor):
    return evidence.anchored_excerpts([{
        'id': 'candidate', 'source_id': 'guide', 'text': anchor,
        'original_excerpt': text,
    }])


@pytest.mark.parametrize('record,anchor', [
    ('Incremental backups are supported.\n\nThey require a matching baseline.',
     'Incremental backups are supported.'),
    ('Available only with a paid licence:\n\n- Journal backups\n- Point-in-time recovery',
     'Journal backups'),
    ('# Aster recovery\nIncremental backups are supported.\n\n'
     '## Conditions\n- They require a matching baseline.',
     'Incremental backups are supported.'),
    ('[### Journal recovery](https://example.test/journal)\n\n'
     'Available from version 4.2.\n\nRequires a matching baseline.',
     'Available from version 4.2.'),
    ('# Recovery limits\nValues are in minutes.\n\n'
     '| Mode | Limit |\n| --- | --- |\n| Journal | 12* |\n\n'
     '*Only when a matching baseline exists.',
     '| Journal | 12* |'),
])
def test_excerpt_retains_complete_governing_context(record, anchor):
    following = '\n\n# Unrelated section\nA different configuration.'
    if record.startswith('[###'):
        following = '\n\n[### Other recovery](https://example.test/other)\nOther facts.'
    excerpts = quote(record + following, anchor)
    assert [item['text'] for item in excerpts] == [record]
    assert excerpts[0]['parent_passage_id'] == 'candidate'


def test_explicit_omissions_never_join_selected_records_or_unselected_tail():
    first = 'Journal backups preserve changes only since a matching baseline.'
    second = 'Snapshot exports require a clean shutdown.'
    tail = 'Preferences and profiling choices can be changed at any time.'
    excerpts = quote(first + '\n[…]\n' + second + '\n[...]\n' + tail, first + '\n' + second)
    assert [item['text'] for item in excerpts] == [first, second]
    assert len({item['id'] for item in excerpts}) == 2
    assert all(item['text'] in first + '\n[…]\n' + second for item in excerpts)


def test_link_navigation_is_unavailable_as_a_qualified_fact():
    text = '# Available categories\nAccording to the index:\n\n- [Backups](https://example.test/a)\n- [Recovery](https://example.test/b)'
    assert quote(text, '[Backups](https://example.test/a)') == []


def test_oversized_section_is_omitted_without_clipping_conditions():
    text = '# Recovery\nJournal backups are supported.\n' + 'Details. ' * 300 + '\nOnly with a matching baseline.'
    assert quote(text, 'Journal backups are supported.') == []


def test_truncated_final_section_cannot_drop_conditions_after_source_limit():
    complete = '# Complete\nSnapshot exports require a clean shutdown.'
    partial = '# Incomplete\nJournal backups are supported.\n'
    text = complete + '\n\n' + partial + 'Detail. ' * 3100 + '\nOnly with a matching baseline.'
    sources = [{
        'source': {'id': 'guide'}, 'document': [text],
        'metadata': [{'source': 'guide', 'research_kind': 'web'}],
    }]
    candidates = evidence.passages(sources)
    incomplete = next(item for item in candidates if 'Journal backups are supported.' in item['text'])
    assert evidence.anchored_excerpts([{**incomplete, 'text': 'Journal backups are supported.'}]) == []
    first = {**candidates[0], 'text': 'Snapshot exports require a clean shutdown.'}
    assert [item['text'] for item in evidence.anchored_excerpts([first])] == [complete]


def test_window_boundary_keeps_original_heading_and_later_qualifier():
    record = '# Recovery\n' + 'Context. ' * 110 + '\nJournal backups are supported.\n\nOnly with a matching baseline.'
    sources = [{
        'source': {'id': 'guide'}, 'document': [record],
        'metadata': [{'source': 'guide', 'research_kind': 'web'}],
    }]
    candidates = evidence.passages(sources)
    selected = next(item for item in candidates if 'Journal backups are supported.' in item['text'])
    assert '# Recovery' not in selected['text']
    excerpts = evidence.anchored_excerpts([{**selected, 'text': 'Journal backups are supported.'}])
    assert [item['text'] for item in excerpts] == [record]
    assert evidence.source_groups(excerpts)[0]['document'] == [record]


def test_nested_fence_text_and_omission_literals_do_not_split_qualifiers():
    record = (
        '# Aster recovery\nBackups are supported.\n````text\n```python\n'
        '# This heading is inside code\n[…]\n```\n````\n'
        'They require a matching baseline.'
    )
    assert [item['text'] for item in quote(record + '\n# Other topic\nOther facts.', 'Backups are supported.')] == [record]
