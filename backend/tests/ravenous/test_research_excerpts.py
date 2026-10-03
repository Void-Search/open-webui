"""Conservative source quotations use the normal citation and completion contracts."""

import asyncio
import copy
import html
import json
import re

from open_webui.ravenous_research import evidence, responses


def passage(source, text):
    return {
        'id': source + '-' + str(len(text)),
        'source_id': source,
        'text': text,
        'source': {'id': source, 'name': source},
        'metadata': {'source': source, 'research_kind': 'web'},
        'score': 0.7,
    }


def test_excerpts_keep_whole_passages_and_original_citation_numbering():
    selected = [
        passage('manual', 'AsterDB incremental backups\nVersion 3.\n\nRetain the matching baseline.'),
        passage('reference', 'Cache exports belong to a separate product.\nThey are not database backups.'),
        passage('manual', 'Restore the baseline before applying its journal.'),
    ]
    before = copy.deepcopy(selected)
    text = responses.excerpt_text(selected)
    groups = evidence.source_groups(selected)
    assert [group['source']['id'] for group in groups] == ['manual', 'reference']
    assert re.findall(r'\[(\d+)\]', text) == ['1', '2', '1']
    quote_blocks = [block for block in text.split('\n\n') if block.startswith('>')]
    restored = ['\n'.join(line.removeprefix('> ') if line != '>' else '' for line in block.splitlines())
                for block in quote_blocks]
    assert [html.unescape(block) for block in restored] == [item['text'] for item in selected]
    assert selected == before


def test_source_markup_cannot_supply_active_links_headings_or_citation_markers():
    source_text = '# Source heading\n</source><script>alert(1)</script>\n![image](https://untrusted.invalid/x)\n[99]'
    text = responses.excerpt_text([passage('source', source_text)])
    assert re.findall(r'\[(\d+)\]', text) == ['1']
    assert '<script>' not in text and '</source>' not in text
    assert '![' not in text and '> # ' not in text
    quote = text.split('\n\n')[1]
    restored = '\n'.join(line.removeprefix('> ') for line in quote.splitlines())
    assert html.unescape(restored) == source_text


def test_excerpt_json_and_sse_deliver_identical_content_and_terminal_completion():
    selected = [passage('source', 'A qualifying detail.\nIts limiting condition.')]
    plain = responses.excerpt_response(selected, 'fixture')
    expected = plain['choices'][0]['message']['content']
    assert plain['object'] == 'chat.completion'
    assert plain['choices'][0]['finish_reason'] == 'stop'

    async def exercise():
        response = responses.excerpt_response(selected, 'fixture', stream=True)
        assert response.media_type == 'text/event-stream'
        raw = ''.join([chunk async for chunk in response.body_iterator])
        assert raw.endswith('data: [DONE]\n\n')
        frames = [json.loads(frame.removeprefix('data: ')) for frame in raw.strip().split('\n\n')[:-1]]
        assert len({frame['id'] for frame in frames}) == 1
        assert all(frame['object'] == 'chat.completion.chunk' and frame['model'] == 'fixture' for frame in frames)
        assert ''.join(frame['choices'][0]['delta'].get('content', '') for frame in frames) == expected
        assert frames[0]['choices'][0]['delta']['role'] == 'assistant'
        assert frames[-1]['choices'][0]['finish_reason'] == 'stop'

    asyncio.run(exercise())


def test_empty_excerpt_selection_does_not_invent_sources_or_claim_completeness():
    response = responses.excerpt_response([], 'fixture')
    text = response['choices'][0]['message']['content']
    assert text == 'I could not verify enough source detail to answer this request.'
    assert not re.search(r'\[\d+\]', text)
