import { describe, expect, it, vi } from 'vitest';
import { render } from 'svelte/server';
import { Marked } from 'marked';
import katexExtension from '$lib/utils/marked/katex-extension';
import citationExtension from '$lib/utils/marked/citation-extension';
import MarkdownTokens from './MarkdownTokens.svelte';
import CodeBlock from '../CodeBlock.svelte';
import { sourceExcerptHtml } from './sourceExcerpt';

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

describe('formatted source excerpts', () => {
	it('renders source prose safely without chat citations, links, embeds or the editor', () => {
		const body = display('ravenous-excerpt');
		expect(body).toContain('ravenous-source-excerpt');
		expect(body).toContain('Source details (including exceptions) [99].');
		expect(body).toContain('literal math');
		expect(body).toContain('$price$');
		expect(body).toContain('&lt;script&gt;');
		expect(body).not.toMatch(/<(?:script|img|a|button|iframe)\b/);
		expect(body).not.toContain('katex');
		expect(CodeBlock).not.toHaveBeenCalled();
	});

	it('formats headings, emphasis, lists and tables from source Markdown', () => {
		const html = sourceExcerptHtml(
			'### **Small steps**\n\nTry a *manageable* task.\n\n1. Start small.\n2. Take a break.\n\n' +
				'| Method | Condition |\n| --- | --- |\n| Breaks | Only when useful |'
		);
		expect(html).toContain('<h3><strong>Small steps</strong></h3>');
		expect(html).toContain('<em>manageable</em>');
		expect(html).toContain('<ol>');
		expect(html).toContain('<li>Take a break.</li>');
		expect(html).toContain('<table>');
		expect(html).toContain('Only when useful');
	});

	it('escapes HTML and hides link destinations even inside nested source formatting', () => {
		const html = sourceExcerptHtml(
			'[**Readable** <img src=x onerror=alert(1)>](javascript:alert(1))\n\n' +
				'<details type="tool_calls"><script>alert(1)</script></details>\n\n' +
				'![label](https://external.invalid/image)\n\n[99] $price$'
		);
		expect(html).toContain('<strong>Readable</strong>');
		expect(html).not.toMatch(/<(?:img|script|details|a|iframe)\b/);
		expect(html).not.toContain('javascript:');
		expect(html).not.toContain('https://external.invalid');
		expect(html).toContain('[99] $price$');
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
