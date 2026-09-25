import { describe, expect, it } from 'vitest';
import { citedSources, outcomeLabel, researchState, sourceLink } from './research';

describe('research activity', () => {
	it('updates each live query and page without repeating rows', () => {
		const entries = [
			{
				action: 'research_progress',
				detail: { stage: 'query', id: 'q1', query: 'Exact user query', status: 'searching' }
			},
			{
				action: 'research_progress',
				detail: {
					stage: 'query',
					id: 'q1',
					query: 'Exact user query',
					status: 'completed',
					result_count: 2
				}
			},
			{
				action: 'research_progress',
				detail: {
					stage: 'page',
					id: 'p1',
					query_ids: ['q1'],
					status: 'failed',
					failure_code: 'fetch_source_http_403'
				}
			}
		];
		const state = researchState(entries);
		expect(state.queries).toHaveLength(1);
		expect(state.queries[0].query).toBe('Exact user query');
		expect(state.pages[0].failure_code).toBe('fetch_source_http_403');
		expect(outcomeLabel(state.pages[0].failure_code)).toBe('Website returned HTTP 403');
	});
	it('uses the final selection after a context adjustment', () => {
		const state = researchState([
			{ action: 'research_complete', report: { sources: [{ id: 'one', selected: true }] } },
			{
				action: 'research_complete',
				report: { sources: [{ id: 'one', selected: false, reasons: ['context_limit'] }] }
			}
		]);
		expect(state.sources[0].selected).toBe(false);
	});
	it('retains concrete engine failures in live and persisted query reports', () => {
		const query = {
			id: 'q1',
			query: 'A general question',
			status: 'failed',
			failure_code: 'discovery_unavailable',
			failure_detail: 'brave: rate limited; duckduckgo: connection failed'
		};
		const progress = { action: 'research_progress', detail: { stage: 'query', ...query } };
		expect(researchState([progress]).queries[0].failure_detail).toBe(query.failure_detail);
		expect(
			researchState([progress, { action: 'research_complete', report: { queries: [query] } }])
				.queries[0].failure_detail
		).toBe(query.failure_detail);
	});
	it('retains provider and fallback information live and after reload', () => {
		const query = {
			id: 'q1',
			query: 'Subject',
			status: 'completed',
			provider: 'SearXNG',
			fallback_reason: 'Brave Search API: access denied; used SearXNG fallback'
		};
		const live = { action: 'research_progress', detail: { stage: 'query', ...query } };
		expect(researchState([live]).queries[0]).toMatchObject(query);
		expect(
			researchState([{ action: 'research_complete', report: { queries: [query] } }]).queries[0]
		).toMatchObject(query);
	});
	it('distinguishes citations from code and ordinary links', () => {
		expect([
			...citedSources('Supported [1], [2, 3]. `[4]` ```array[5]``` [6](https://example.org)')
		]).toEqual([1, 2, 3]);
	});
	it('allows source links without executable URL schemes', () => {
		expect(sourceLink('https://example.org/source')).toBeTruthy();
		expect(sourceLink('/api/v1/files/one/content')).toBeTruthy();
		expect(sourceLink('javascript:alert(1)')).toBeNull();
		expect(sourceLink('//external.example/file')).toBeNull();
	});
});
