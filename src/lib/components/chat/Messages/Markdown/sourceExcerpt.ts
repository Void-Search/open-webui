import { Marked } from 'marked';

const escapeHtml = (text: string) =>
	text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

// Source Markdown uses its own parser: no chat citations, math, embeds or tools.
// Raw HTML is escaped, and source links/images display only their labels.
const parser = new Marked({
	gfm: true,
	async: false,
	renderer: {
		html: (text) => escapeHtml(text),
		link: (_href, _title, text) => text,
		image: (_href, _title, text) => escapeHtml(text),
		checkbox: (checked) => (checked ? '☑ ' : '☐ ')
	}
});

export const sourceExcerptHtml = (text: string): string => parser.parse(text) as string;
