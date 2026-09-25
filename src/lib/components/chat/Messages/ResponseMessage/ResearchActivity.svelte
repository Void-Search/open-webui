<script lang="ts">
	import { citedSources, outcomeLabel, researchState, sourceLink } from './research';
	import ResearchRecovery from './ResearchRecovery.svelte';
	export let entries: any[] = [];
	export let content = '';
	export let messageId: string;
	export let done = false;
	export let active = true;
	export let readOnly = false;
	export let submit: (reply: string) => Promise<void>;
	$: state = researchState(entries);
	$: cited = citedSources(content);
	const stages: Record<string, string> = {
		planning: 'Planning searches',
		query: 'Searching the web',
		page: 'Reading web pages',
		local: 'Searching local knowledge',
		reranking: 'Reranking web and local evidence',
		verification: 'Checking supporting evidence'
	};
</script>

<div class="my-2 space-y-2 text-sm" data-testid="research-activity">
	<p aria-live="polite" class="text-gray-600 dark:text-gray-400">
		{state.completed
			? state.report?.answer_basis === 'general_knowledge'
				? 'General-knowledge answer · no verified sources'
				: `Research complete · ${state.sources.filter((s: any) => s.selected).length} sources selected`
			: (stages[state.stage] ?? 'Research in progress')}
	</p>
	{#each state.queries as query (query.id)}
		<details class="rounded-lg border border-gray-200 dark:border-gray-800 p-2">
			<summary class="cursor-pointer break-words">
				<span>{query.query}</span>
				<span class="ml-2 text-xs text-gray-500"
					>{outcomeLabel(query.status)} · {query.result_count ?? 0} results</span
				>
			</summary>
			{#if query.provider}<p class="mt-2 text-xs">Search provider: {query.provider}</p>{/if}
			{#if query.fallback_reason}<p class="mt-1 text-xs">{query.fallback_reason}</p>{/if}
			{#if query.failure_code}<p class="mt-2">{outcomeLabel(query.failure_code)}</p>{/if}
			{#if query.failure_detail}<p class="mt-1 text-xs">{query.failure_detail}</p>{/if}
			<ul class="space-y-3 pt-2">
				{#each state.pages.filter( (page: any) => page.query_ids?.includes(query.id) ) as page (page.id)}
					<li>
						{#if sourceLink(page.url)}<a
								class="underline"
								href={sourceLink(page.url)}
								target="_blank"
								rel="noopener noreferrer">{page.title || page.url}</a
							>{:else}<span>{page.title}</span>{/if}
						<p class="text-xs text-gray-500">{outcomeLabel(page.failure_code || page.status)}</p>
						{#if page.snippet}<p class="mt-1 text-gray-600 dark:text-gray-400">
								{page.snippet}
							</p>{/if}
					</li>
				{/each}
			</ul>
		</details>
	{/each}
	{#each Object.entries(state.local) as [name, outcome]}
		<p class="text-gray-600 dark:text-gray-400">
			{name === 'saved' ? 'Saved research' : 'Local knowledge'}: {outcomeLabel(
				(outcome as any).status
			)}{(outcome as any).count !== undefined ? ` · ${(outcome as any).count} sources` : ''}
		</p>
	{/each}
	{#if state.completed}
		<details class="rounded-lg border border-gray-200 dark:border-gray-800 p-2">
			<summary class="cursor-pointer">Sources and retrieval details</summary>
			<p class="my-2 text-gray-600 dark:text-gray-400">{state.report.summary}</p>
			<ul class="space-y-2">
				{#each state.sources as source (source.id)}
					<li>
						{#if sourceLink(source.url)}<a
								class="underline"
								href={sourceLink(source.url)}
								target="_blank"
								rel="noopener noreferrer">{source.title}</a
							>{:else}<span>{source.title}</span>{/if}
						<span class="ml-1 text-xs text-gray-500"
							>{source.kind} · {source.selected
								? cited.has(source.citation)
									? 'Selected · Cited'
									: 'Selected for generation'
								: 'Retrieved · Excluded'}</span
						>
						{#if !source.selected}<p class="text-xs text-gray-500">
								{source.reasons.map(outcomeLabel).join(', ')}
							</p>{/if}
					</li>
				{/each}
			</ul>
		</details>
	{/if}
	<ResearchRecovery {messageId} recovery={state.recovery} {done} {active} {readOnly} {submit} />
</div>
