<script lang="ts">
	import { setContext } from 'svelte';
	import { writable } from 'svelte/store';
	setContext('i18n', writable({ t: (value: string) => value }));
	import ResearchActivity from '../../../src/lib/components/chat/Messages/ResponseMessage/ResearchActivity.svelte';
	const saved = JSON.parse(sessionStorage.getItem('research-ui-fixture') || 'null');
	let messageId = saved?.messageId || `fixture-${Date.now()}`;
	let done = saved?.done || false;
	let active = true;
	let entries: any[] = saved?.entries || [];
	let replies: string[] = [];
	const queries = Array.from({ length: 5 }, (_, i) => ({
		id: `q${i}`,
		query: `Complementary search ${i + 1}`,
		status: 'completed',
		result_count: 1
	}));
	const pages = queries.map((query, i) => ({
		id: `p${i}`,
		query_ids: [query.id],
		url: `https://example.org/${i}`,
		title: `Source ${i + 1}`,
		snippet: 'Discovery snippet',
		status: i === 2 ? 'failed' : 'read',
		failure_code: i === 2 ? 'fetch_source_http_403' : null
	}));
	function start() {
		messageId = `fixture-${Date.now()}`;
		done = false;
		active = true;
		replies = [];
		entries = queries.map((query) => ({
			action: 'research_progress',
			detail: { stage: 'query', ...query }
		}));
		entries = [
			...entries,
			...pages.map((page) => ({ action: 'research_progress', detail: { stage: 'page', ...page } }))
		];
		sessionStorage.removeItem('research-ui-fixture');
	}
	function finish() {
		entries = [
			...entries,
			{
				action: 'research_complete',
				report: {
					queries,
					pages,
					local: {
						native: { status: 'completed', count: 1 },
						saved: { status: 'completed', count: 0 }
					},
					summary:
						'Five searches completed. One website returned HTTP 403. Local retrieval found one source.',
					sources: [
						{
							id: 'web',
							title: 'Public reference',
							kind: 'web',
							selected: true,
							citation: 1,
							reasons: []
						},
						{
							id: 'local',
							title: 'Local manual',
							kind: 'local',
							selected: true,
							citation: 2,
							reasons: []
						},
						{
							id: 'extra',
							title: 'Repeated content',
							kind: 'web',
							selected: false,
							reasons: ['duplicate']
						}
					]
				},
				recovery: {
					kind: 'clarification',
					question: 'Which edition should I check?',
					choices: [
						{ label: 'Current edition', reply: 'Current edition', mode: 'both' },
						{ label: 'Previous edition', reply: 'Previous edition', mode: 'both' }
					]
				},
				done: true
			}
		];
		done = true;
		sessionStorage.setItem('research-ui-fixture', JSON.stringify({ messageId, done, entries }));
	}
	async function submit(reply: string) {
		replies = [...replies, reply];
	}
</script>

<main style="max-width: 800px; margin: 30px auto; font-family: sans-serif">
	<h1>Research UI acceptance</h1>
	<button on:click={start}>Start search</button>
	<button on:click={finish}>Finish research</button>
	<button on:click={() => (active = !active)}>Switch branch</button>
	{#key messageId}
		<ResearchActivity
			{entries}
			{messageId}
			{done}
			{active}
			{submit}
			content="Supported facts [1] and [2]."
		/>
	{/key}
	<p data-testid="submitted">{JSON.stringify(replies)}</p>
</main>
