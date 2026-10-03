import { describe, expect, it, vi } from 'vitest';
import { render } from 'svelte/server';
import { Marked } from 'marked';
import katexExtension from '$lib/utils/marked/katex-extension';
import citationExtension from '$lib/utils/marked/citation-extension';
import MarkdownTokens from './MarkdownTokens.svelte';
import CodeBlock from '../CodeBlock.svelte';

vi.mock('../CodeBlock.svelte', () => ({ default: vi.fn() }));

const marked = new Marked(katexExtension(), citationExtension());
const excerpt = String.raw`Source details (including exceptions) [99].
    Keep indentation and \(literal math\), \[99\], $price$.
<script>alert('source')</script> ![image](https://example.invalid/x)

\*Source punctuation\* and \\ remain literal.`;

function display(language: string, legacyResearchExcerpts = false, quote = true) {
	const fence = '````';
	let content = `${fence}${language}\n${excerpt}\n${fence}`;
	if (quote)
		content = content
			.split('\n')
			.map((line) => `> ${line}`)
			.join('\n');
	vi.mocked(CodeBlock).mockClear();
	return render(MarkdownTokens, {
		props: { id: 'excerpt', tokens: marked.lexer(content), legacyResearchExcerpts }
	}).body;
}

describe('literal source excerpts', () => {
	it('renders source text as escaped prose without the code editor', () => {
		const body = display('ravenous-excerpt');
		expect(body).toContain('ravenous-source-excerpt whitespace-pre-wrap');
		expect(body).toContain('Source details (including exceptions) [99].');
		expect(body).toContain(String.raw`    Keep indentation and \(literal math\), \[99\], $price$.`);
		expect(body).toContain('&lt;script>');
		expect(body).not.toMatch(/<(?:script|img|a|button|pre|code)\b/);
		expect(CodeBlock).not.toHaveBeenCalled();
	});

	it('restyles saved text excerpts only inside flagged research blockquotes', () => {
		expect(display('text', true)).toContain('ravenous-source-excerpt');
		expect(CodeBlock).not.toHaveBeenCalled();
		for (const [review, quote] of [
			[false, true],
			[true, false],
			[false, false]
		]) {
			expect(display('text', review, quote)).not.toContain('ravenous-source-excerpt');
			expect(CodeBlock).toHaveBeenCalledOnce();
		}
		expect(display('python', true)).not.toContain('ravenous-source-excerpt');
		expect(CodeBlock).toHaveBeenCalledOnce();
	});
});
