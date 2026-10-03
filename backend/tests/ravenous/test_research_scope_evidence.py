"""Verification and conservative output use the same complete qualified evidence."""

import asyncio

from open_webui.ravenous_research import evidence, pipeline


def candidate(text, identity='manual'):
    return evidence.passages([{
        'source': {'id': identity, 'name': 'Qualified source'},
        'document': [text],
        'metadata': [{'source': identity, 'research_kind': 'web'}],
    }])


def research():
    value = pipeline.NativeResearch(None, {'model': 'fixture'}, {'__event_emitter__': None}, None)
    value.metadata['ravenous_review_previous_answer'] = True
    value.question = 'Explain the journal option for AsterDB version 3.'
    value.user_context = 'What about the journal option?'
    return value


def test_review_sends_complete_sections_once_with_their_source_scope():
    text = '# AsterDB 3 journal backups\n' + 'Context. ' * 110 + '\nAvailable online.\n\nRequires the matching baseline.'
    fitted = candidate(text)
    assert len(fitted) > 1
    refs, data = research().assessment_input(fitted)
    assert data['evidence_unit'] == 'complete_source_section'
    assert len(refs) == len(data['passages']) == 1
    assert data['passages'][0]['text'] == text
    assert data['sources'][0]['title'] == 'Qualified source'


def test_selected_section_cannot_expand_into_rejected_sections_with_shared_text(monkeypatch):
    wanted = '# AsterDB version 3\nJournal backups are supported.\nA matching baseline is required.'
    rejected = '# AsterCache version 4\nJournal backups are supported.\nA separate cache product.'
    fitted = candidate(wanted + '\n\n' + rejected)
    value = research()

    async def choose(_request, _model, _user, _prompt, data, **_kwargs):
        assert [p['text'] for p in data['passages']] == [wanted, rejected]
        return {'sufficient': True, 'supported_ids': [data['passages'][0]['id']],
                'missing': [], 'conflicts': [], 'choices': [], 'clarification_question': None}

    monkeypatch.setattr(pipeline, 'model_json', choose)
    assessed = asyncio.run(value.assess(fitted))
    selected = [{**item, 'text': assessed['supported_text'][item['id']],
                 'verified_sections': assessed['supported_sections'][item['id']]}
                for item in fitted if item['id'] in assessed['supported_ids']]
    quotes = evidence.anchored_excerpts(selected)
    assert [item['text'] for item in quotes] == [wanted]
    assert 'AsterCache' not in evidence.source_groups(quotes)[0]['document'][0]


def test_unselected_sections_cannot_be_reintroduced_after_empty_assessment(monkeypatch):
    fitted = candidate('# Another product\nA different version and deployment region.')
    value = research()

    async def reject(*_args, **_kwargs):
        return {'sufficient': False, 'supported_ids': [], 'missing': ['The requested product is unverified.'],
                'conflicts': [], 'choices': [], 'clarification_question': None}

    monkeypatch.setattr(pipeline, 'model_json', reject)
    assessed = asyncio.run(value.assess(fitted))
    assert assessed['supported_sections'] == assessed['supported_text'] == {}
    assert evidence.anchored_excerpts([{**fitted[0], 'verified_sections': []}]) == []


def test_navigation_and_incomplete_sections_cannot_be_verification_units():
    value = research()
    navigation = candidate('# Categories\n- [Journal](https://example.test/journal)\n- [Snapshot](https://example.test/snapshot)')
    oversized = candidate('# Scope\n' + 'Details. ' * 300 + '\nOnly with a baseline.')
    refs, data = value.assessment_input([*navigation, *oversized])
    assert refs == {} and data['passages'] == []
