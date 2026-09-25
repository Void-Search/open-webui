export function researchState(entries: any[]) {
	const queries = new Map<string, any>();
	const pages = new Map<string, any>();
	const local = new Map<string, any>();
	let stage = 'planning';
	let completed: any = null;
	for (const entry of entries) {
		if (entry.action === 'research_complete') completed = entry;
		if (entry.action !== 'research_progress') continue;
		const detail = entry.detail ?? {};
		stage = detail.stage ?? stage;
		if (stage === 'query') queries.set(detail.id, detail);
		if (stage === 'page') pages.set(detail.id, detail);
		if (stage === 'local') local.set(detail.store, detail);
	}
	return {
		queries: completed?.report?.queries ?? [...queries.values()],
		pages: completed?.report?.pages ?? [...pages.values()],
		local: completed?.report?.local ?? Object.fromEntries(local),
		sources: completed?.report?.sources ?? [],
		report: completed?.report,
		recovery: completed?.recovery,
		stage,
		completed: Boolean(completed)
	};
}

export function citedSources(text: string): Set<number> {
	const citations = new Set<number>();
	// Exclude code: bracketed array indexes are not citations.
	const prose = text.replace(/```[\s\S]*?```/g, '').replace(/`[^`]*`/g, '');
	for (const match of prose.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g)) {
		for (const value of match[1].split(',')) citations.add(Number(value.trim()));
	}
	return citations;
}

export function sourceLink(value: unknown): string | null {
	if (typeof value !== 'string') return null;
	if (
		/^https?:\/\//i.test(value) ||
		/^\/api\/v1\/(?:files|ravenous\/research\/knowledge)\//.test(value)
	)
		return value;
	return null;
}

export function outcomeLabel(value: string): string {
	const labels: Record<string, string> = {
		duplicate: 'Duplicate content',
		low_relevance: 'Low relevance',
		context_limit: 'Context limit',
		candidate_limit: 'Candidate pool limit',
		access_revoked: 'Access no longer available',
		unsupported: 'Did not support the requested facts',
		verification_incomplete: 'Evidence verification did not complete',
		page_limit: 'Page limit',
		source_outside_scope: 'Outside requested sources',
		already_attempted: 'Already attempted',
		fetch_timeout: 'Page reading timed out',
		discovery_timeout: 'Search timed out',
		fetch_empty: 'No readable text',
		fetch_failed: 'Extraction failed',
		response_too_large: 'Page too large',
		discovery_unavailable: 'Search service unavailable',
		fetch_unavailable: 'Page-reading service unavailable',
		no_results: 'No search results',
		deadline: 'Time limit reached',
		read: 'Read',
		reading: 'Reading',
		discovered: 'Found',
		searching: 'Searching',
		completed: 'Completed',
		failed: 'Failed',
		skipped: 'Skipped',
		partial: 'Partially completed'
	};
	if (value?.startsWith('fetch_source_http_'))
		return `Website returned HTTP ${value.split('_').at(-1)}`;
	if (value?.startsWith('fetch_http_'))
		return `Reading service returned HTTP ${value.split('_').at(-1)}`;
	return labels[value] ?? value;
}
