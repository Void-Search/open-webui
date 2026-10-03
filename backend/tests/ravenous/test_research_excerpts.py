"""Conservative source quotations use the normal citation and completion contracts."""

import asyncio
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from open_webui.ravenous_research import context, conversation, evidence, responses


def passage(source, text):
    return {
        'id': source + '-' + str(len(text)),
        'source_id': source,
        'text': text,
        'source': {'id': source, 'name': source},
        'metadata': {'source': source, 'research_kind': 'web'},
        'score': 0.7,
    }



def quoted_text(block):
    lines = [line.removeprefix('> ') if line != '>' else '' for line in block.splitlines()]
    fence = lines[0].removesuffix('text')
    assert len(fence) >= 3 and set(fence) == {'`'} and lines[-1] == fence
    return '\n'.join(lines[1:-1])


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
    assert [quoted_text(block) for block in quote_blocks] == [item['text'] for item in selected]
    assert selected == before


def test_source_markup_cannot_supply_active_links_headings_or_citation_markers():
    source_text = '# Source heading\n</source><script>alert(1)</script>\n![image](https://untrusted.invalid/x)\n[99]'
    text = responses.excerpt_text([passage('source', source_text)])
    quote = text.split('\n\n')[1]
    assert quoted_text(quote) == source_text.replace('![image](https://untrusted.invalid/x)', 'image')
    assert text.split('\n\n')[-1] == '[1]'
    assert 'https' not in text and '&#' not in text


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


def test_link_labels_retain_factual_text_without_long_destinations():
    original = '[Version 3 restore](https://example.org/manual_(version_3)?tracking=long)\nKeep its baseline.\n\n[99] is source text.'
    text = responses.excerpt_text([passage('manual', original)])
    assert 'https' not in text and 'tracking' not in text
    quote = text.split('\n\n')[1]
    assert quoted_text(quote) == 'Version 3 restore\nKeep its baseline.\n\n[99] is source text.'
    assert text.split('\n\n')[-1] == '[1]'


def test_webui_tokenizer_renders_readable_quotes_without_active_source_markup():
    fork = Path(__file__).parents[3]
    if not shutil.which('node') or not all((fork / 'node_modules' / name).is_dir()
                                        for name in ('marked', 'typescript')):
        pytest.skip('WebUI Node dependencies are required for its actual Markdown tokenizer')
    original = (
        '# Heading\n[Readable label](https://untrusted.invalid/long/path) & a condition.\n'
        '[99] <script>x</script> ![photo](https://untrusted.invalid/image)\n\n'
        '    Indented text (such as an exception) [remains separate].\n'
        r'Literal \(math-looking text\) and \[99\] and $price$.' + '\n'
        '````\nA source fence must stay quoted.\n````'
    )
    text = responses.excerpt_text([passage('source', original)])
    script = r"""
import fs from 'node:fs';
import { Marked } from 'marked';
import ts from 'typescript';
async function extension(name) {
  const source = fs.readFileSync('src/lib/utils/marked/' + name + '-extension.ts', 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
  return (await import('data:text/javascript;base64,' + Buffer.from(compiled).toString('base64'))).default;
}
const katex = await extension('katex');
const citation = await extension('citation');
const marked = new Marked(katex(), citation());
const tokens = marked.lexer(fs.readFileSync(0, 'utf8'));
const found = [];
function inspect(items) {
  for (const item of items) {
    found.push({ type: item.type, ids: item.ids, lang: item.lang });
    if (item.tokens) inspect(item.tokens);
  }
}
inspect(tokens);
console.log(JSON.stringify({ found, quotes: tokens.filter(item => item.type === 'blockquote').map(item => item.tokens.map(part => part.text).join('')) }));
"""
    result = subprocess.run(['node', '--input-type=module', '-e', script], input=text, text=True,
                            cwd=fork, capture_output=True, check=True)
    rendered = json.loads(result.stdout)
    expected = original.replace('[Readable label](https://untrusted.invalid/long/path)', 'Readable label')
    expected = expected.replace('![photo](https://untrusted.invalid/image)', 'photo')
    assert rendered['quotes'] == [expected]
    assert [token['ids'] for token in rendered['found'] if token['type'] == 'citation'] == [[1]]
    assert [token.get('lang') for token in rendered['found'] if token['type'] == 'code'] == ['text']
    assert not any(token['type'] in ('html', 'link', 'image', 'heading', 'inlineKatex', 'blockKatex')
                   for token in rendered['found'])


def test_review_context_budget_loss_keeps_empty_excerpt_response_source_limited(monkeypatch):
    selected = [passage('manual', 'Journal backups\nVersion 3 only.\nRetain the complete baseline.')]
    original = evidence.context_message(selected, 'What did the earlier answer miss?')['content']
    metadata = {
        'ravenous_review_previous_answer': True,
        'ravenous_selected_passages': selected,
        'ravenous_evidence_message': original,
        'ravenous_research_context': {'query': 'What did the earlier answer miss?'},
        'ravenous_research': {'report': {
            'sources': evidence.source_report(selected, selected), 'summary': 'Retrieval completed.',
        }},
    }
    body = {'model': 'fixture', 'metadata': metadata, 'messages': [{'role': 'system', 'content': original}]}
    misc = ModuleType('open_webui.utils.misc')
    misc.merge_system_messages = lambda messages: messages
    misc.strip_empty_content_blocks = lambda messages: messages
    monkeypatch.setitem(sys.modules, misc.__name__, misc)
    counted = []

    async def fit(_request, _body, _user, candidates, _question):
        counted.extend(candidates)
        assert candidates[0]['text'] == selected[0]['text']
        return [], []

    async def remember(data, _user, result):
        data['ravenous_research'] = result

    monkeypatch.setattr(context, 'fit', fit)
    monkeypatch.setattr(conversation, 'remember_result', remember)
    sources = asyncio.run(context.finalize(None, body, None, None))
    assert counted and sources == [] and metadata['ravenous_selected_passages'] == []
    state = metadata['ravenous_research']
    assert state['answer_basis'] == state['report']['answer_basis'] == 'source_limited'
    assert 'general knowledge' not in state['report']['summary']
    response = responses.excerpt_response(metadata['ravenous_selected_passages'], 'fixture')
    assert response['choices'][0]['message']['content'] == 'I could not verify enough source detail to answer this request.'
