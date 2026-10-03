"""Web furniture is excluded before selection without losing advice or conditions."""

import pytest
from open_webui.ravenous_research import evidence
from open_webui.ravenous_research.source_text import clean_web_text


def source(text, kind='web'):
    return {'source': {'id': 'https://example.test/advice'}, 'document': [text],
            'metadata': [{'research_kind': kind, 'source': 'https://example.test/advice'}]}


def test_reported_breadcrumbs_ads_and_footer_leave_article_advice():
    text = ('[Home](https://example.test/)\n\n / [Articles](https://example.test/articles/)\n'
            '\n / General Productivity\n\n### Set manageable goals\n'
            'Break large tasks into smaller steps.\n\nAdvertisement\n\n'
            'Allow flexibility rather than scheduling every minute.\n\nAbout the author\n'
            '\nChris Example\n\nRelated articles\n\nhttps://example.test/unrelated\n'
            'A different article teaser.')
    candidates = evidence.passages([source(text)])
    selected = evidence.anchored_excerpts(candidates)
    assert [item['text'] for item in selected] == [
        '### Set manageable goals\nBreak large tasks into smaller steps.\n\n'
        'Allow flexibility rather than scheduling every minute.'
    ]
    assert all(item['text'] in clean_web_text(text) for item in selected)


def test_removing_an_ad_never_detaches_governing_qualifiers():
    text = ('## Archive recovery\nAvailable only with a paid licence:\n\n'
            'Advertisement\n\n- Journal backups\n- Point-in-time recovery\n\n'
            'Requires a matching baseline.\n\n### Exceptions\n'
            'Not available for cache exports.')
    selected = evidence.anchored_excerpts(evidence.passages([source(text)]))
    assert len(selected) == 1
    assert all(fragment in selected[0]['text'] for fragment in (
        'Available only with a paid licence:', '- Journal backups',
        'Requires a matching baseline.', 'Not available for cache exports.',
    ))


def test_offers_are_removed_without_rewriting_advice_or_paid_conditions():
    text = ('## Small steps\nWrite down the next manageable step.\n\n'
            'In my course you can buy our printable and join us.\n\n'
            'The course requires a paid licence. Free examples are unavailable.\n\n'
            'This method is useful only when the task is clearly defined.')
    cleaned = clean_web_text(text)
    assert 'buy our printable' not in cleaned
    assert 'The course requires a paid licence. Free examples are unavailable.' in cleaned
    assert 'This method is useful only when the task is clearly defined.' in cleaned


def test_footer_does_not_swallow_a_later_article_section():
    text = ('## Advice\nTake a small first step.\n\n## Related articles\n'
            '[Other advice](https://example.test/other)\n\n## Conditions\n'
            'Stop if the situation becomes unsafe.')
    cleaned = clean_web_text(text)
    assert 'Related articles' not in cleaned and 'Other advice' not in cleaned
    assert 'Stop if the situation becomes unsafe.' in cleaned


def test_offer_display_labels_and_terminal_taglines_are_not_advice():
    text = ('## Small goals\nBreak tasks into steps.\n\nOnly with a realistic goal\n\n'
            '10\n\nTask\n_Busters_\n\nExample Author\n\nFree printable\n'
            '\nSend me the printable\n\nAbout the author')
    cleaned = clean_web_text(text)
    assert cleaned == '## Small goals\nBreak tasks into steps.\n\nOnly with a realistic goal'
    assert clean_web_text('## Small steps\nStart small.\n\nBetter work starts here') == '## Small steps\nStart small.'


def test_code_examples_and_local_documents_are_not_treated_as_web_ui():
    text = '## Interface labels\n```text\nAdvertisement\nSubscribe\n / Articles\n```\nThese labels are required.'
    assert clean_web_text(text) == text
    local = 'Advertisement\n\nHome\n / Articles\nThe literal labels must be copied.'
    assert any('Advertisement' in item['text'] for item in evidence.passages([source(local, 'local')]))


@pytest.mark.parametrize('fence', ['```', '~~~~'])
def test_literal_offer_strings_and_blank_lines_in_code_are_preserved(fence):
    text = ('## Message templates\n' + fence + 'text\n'
            'template = "Buy my course and join us."\n\n\n'
            'message = "Subscribe to our newsletter"\n' + fence + '\n'
            'These literal strings must be copied unchanged.\n\n'
            'Buy our course and join us.')
    assert clean_web_text(text) == text.rsplit('\n\n', 1)[0]


def test_article_masthead_labels_are_removed_without_cutting_conditions():
    text = ('Mechanical and Biomedical Engineering\n\nNews\n\n'
            'Most of us procrastinate.\n\n## Start small\n'
            'Pick one manageable task.\n\nOnly with a realistic goal')
    cleaned = clean_web_text(text)
    assert cleaned.startswith('Most of us procrastinate.')
    assert 'News' not in cleaned and 'Mechanical' not in cleaned
    assert cleaned.endswith('Only with a realistic goal')
    assert clean_web_text('\n\n' + text) == cleaned
    assert clean_web_text('Only with permission\n\nBefore beginning\n\nProceed carefully.') == (
        'Only with permission\n\nBefore beginning\n\nProceed carefully.')
    assert clean_web_text('2026\n\nNews\n\nThis report applies to the stated year.') == (
        '2026\n\nNews\n\nThis report applies to the stated year.')
    assert clean_web_text('Free plan\n\nPreview mode\n\nThese constraints apply.') == (
        'Free plan\n\nPreview mode\n\nThese constraints apply.')


def test_unrelated_link_card_cannot_be_quoted_as_article_body():
    card = ('https://example.test/different-article\n\n'
            'A teaser for another article about an unrelated condition.\n\nHealth categories')
    candidates = evidence.passages([source(card)])
    assert evidence.anchored_excerpts(candidates) == []


def test_linked_headings_and_source_identity_are_not_rejected_as_cards():
    for text in (
        '[### Recovery](https://example.test/recovery)\nRequires a matching baseline.',
        'https://example.test/advice\nThis article explains how to take a small first step.',
    ):
        assert evidence.anchored_excerpts(evidence.passages([source(text)]))


def test_directory_controls_do_not_replace_the_requested_details():
    text = ('Create Event\n\nOpen App\n\n## Willow Hall\n'
            'Address: 17 Orchard Road, Oxford.\n\nCheck ticket price on event\n\n'
            '_Save this event: A talk__Share this event: A talk_\n\nPrevious\n\nExplore more events')
    assert clean_web_text(text) == '## Willow Hall\nAddress: 17 Orchard Road, Oxford.'
    literal = '## Labels\n```text\nCreate Event\nSave this event: A talk\n```\nUse these exact labels.'
    assert clean_web_text(literal) == literal


def test_named_details_exclude_unrelated_directory_entries_but_keep_source_identity():
    def item(text, title='Events in Oxford'):
        return {'text': text, 'source_id': 'directory', 'source': {'name': title},
                'metadata': {'research_kind': 'web'}}

    assert evidence.quotation_records(item('## Cedar Centre\nAddress: 2 School Road.'), ['Willow Hall']) == []
    assert evidence.quotation_records(item('## Willow Hall\nAddress: 17 Orchard Road.'), ['Willow Hall'])
    assert evidence.quotation_records(item('## Contact\n17 Orchard Road.', 'Willow Hall – Contact'), ['Willow Hall'])
    assert evidence.quotation_records(item('## Willow Hall\n17 Orchard Road, Oxford.'), ['Willow Hall, Oxford'])
    assert evidence.quotation_records(item('## Willowy Hallway\n17 Orchard Road.'), ['Willow Hall']) == []
