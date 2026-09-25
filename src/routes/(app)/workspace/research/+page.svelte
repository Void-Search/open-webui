<script lang="ts">
	import { onMount } from 'svelte';
	import { v4 as uuidv4 } from 'uuid';
	let documents: any[] = [];
	let collections: any[] = [];
	let selected: string[] = [];
	let collection = '';
	let name = '';
	let principal = '';
	let shareId = '';
	let source = '';
	let error = '';
	let busy = false;
	let mode = 'auto';
	let question = '';
	let selectedOnly = false;
	let messages: { role: string; content: string }[] = [];
	let answers: any[] = [];
	function retainedLabel(answer: any, index: number) {
        const temporary = new Set((answer.research?.web_sources ?? []).map((item: any) => item.source_id));
        let remaining = index;
        for (let label = 1; ; label++) {
            if (!temporary.has(`S${label}`)) {
                if (remaining === 0) return `S${label}`;
                remaining--;
            }
        }
    }
	async function ask() {
		await action(async () => {
			const pending = [...messages, { role: 'user', content: question }];
			const response = await fetch('/api/v1/ravenous/research/chat', {
				method: 'POST',
				headers: {
					Authorization: `Bearer ${localStorage.token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					messages: pending,
					knowledge: { document_ids: selected, collection_ids: collection ? [collection] : [] },
					research: { mode, selected_sources_only: selectedOnly }
				})
			});
			if (!response.ok) throw new Error(`Research answer failed (${response.status}).`);
			const answer = (await response.json()).choices[0];
			messages = [...pending, answer.message];
			answers = [...answers, { question, ...answer }];
			question = '';
		});
	}
	const base = '/api/v1/ravenous/research/knowledge';
	async function api(
		path: string,
		method = 'GET',
		body?: any,
		headers: Record<string, string> = {}
	) {
		const response = await fetch(base + path, {
			method,
			headers: { Authorization: `Bearer ${localStorage.token}`, ...headers },
			body: body instanceof File ? body : body === undefined ? undefined : JSON.stringify(body)
		});
		if (!response.ok) throw new Error(`Research request failed (${response.status}).`);
		return response.json();
	}
	async function action(work: () => Promise<void>) {
		busy = true;
		error = '';
		try {
			await work();
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
		}
	}
	async function refresh() {
		documents = (await api('/documents')).data;
		collections = (await api('/collections')).data;
	}
	async function upload(event: Event) {
		const file = (event.target as HTMLInputElement).files?.[0];
		if (!file) return;
		await action(async () => {
			const result = await api('/documents', 'POST', file, {
				'Content-Type': file.type || 'application/octet-stream',
				'X-Filename': encodeURIComponent(file.name),
				'Idempotency-Key': uuidv4()
			});
			if (collection)
				await api(`/collections/${collection}/documents/${result.document_id}`, 'PUT');
			await refresh();
		});
	}
	onMount(() => {
		action(refresh);
	});
</script>

<div class="max-w-4xl mx-auto p-4 space-y-5">
	<h1 class="text-xl font-semibold">Research knowledge</h1>
	<p>
		Uploads are stored in independent knowledge. <a class="underline" href="/workspace/knowledge"
			>Open legacy knowledge</a
		>.
	</p>
	{#if error}<p role="alert">{error}</p>{/if}
	<div class="flex gap-3 flex-wrap">
		<label>Upload document <input type="file" disabled={busy} on:change={upload} /></label>
		<button disabled={busy} on:click={() => action(refresh)}>Refresh status</button>
	</div>
	<label
		>Collection
		<select bind:value={collection}
			><option value="">None</option>{#each collections as item}<option value={item.collection_id}
					>{item.name}</option
				>{/each}</select
		>
	</label>
	<div class="flex gap-2">
		<input aria-label="New collection name" placeholder="Collection name" bind:value={name} />
		<button
			disabled={busy || !name}
			on:click={() =>
				action(async () => {
					await api(
						'/collections',
						'POST',
						{ name, document_ids: selected },
						{ 'Content-Type': 'application/json' }
					);
					name = '';
					await refresh();
				})}>Create collection from selection</button
		>
	</div>
	<ul class="space-y-2">
		{#each documents as document}
			<li class="flex gap-3 items-center">
				<input
					type="checkbox"
					aria-label={`Select ${document.title}`}
					bind:group={selected}
					value={document.document_id}
				/>
				<span>{document.title} — {document.status}</span>
				<button
					disabled={busy}
					on:click={() =>
						action(async () => {
							source = (
								await api(
									`/documents/${document.document_id}/download?revision_id=${document.revision_id}`
								)
							).content;
						})}>View source</button
				>
			</li>
		{/each}
	</ul>
	<div class="space-y-2">
		<p>Share the selected collection, or the first selected document, with a stable principal.</p>
		<input aria-label="Principal ID" placeholder="Principal ID" bind:value={principal} />
		<button
			disabled={busy || !principal || (!collection && !selected.length)}
			on:click={() =>
				action(async () => {
					const target = collection ? `/collections/${collection}` : `/documents/${selected[0]}`;
					shareId = (
						await api(
							target + '/shares',
							'POST',
							{ principal_id: principal, scopes: ['knowledge:read'] },
							{ 'Content-Type': 'application/json' }
						)
					).share_id;
				})}>Grant read access</button
		>
		<input aria-label="Share ID to revoke" placeholder="Share ID" bind:value={shareId} />
		<button
			disabled={busy || !shareId || (!collection && !selected.length)}
			on:click={() =>
				action(async () => {
					const target = collection ? `/collections/${collection}` : `/documents/${selected[0]}`;
					await api(target + '/shares/' + shareId, 'DELETE');
					shareId = '';
				})}>Revoke access</button
		>
	</div>
	<section class="space-y-3 border-t pt-4">
		<h2 class="text-lg font-semibold">Research chat</h2>
		<label
			>Research mode <select bind:value={mode}
				><option value="private">PRIVATE — saved sources</option><option value="auto"
					>AUTO — saved sources and web</option
				></select
			></label
		>
		<label><input type="checkbox" bind:checked={selectedOnly} /> Use selected sources only</label>
		{#each answers as answer}
			<article class="space-y-2 border rounded p-3">
				<p class="font-semibold">{answer.question}</p>
				<p class="whitespace-pre-wrap">{answer.message.content}</p>
				{#each answer.research?.notices ?? [] as notice}
					<p role="status">
						{notice === 'snapshot_not_saved'
							? 'Snapshot not saved. This answer uses a temporary web capture.'
							: notice.replaceAll('_', ' ')}
					</p>
				{/each}
				{#each answer.sources ?? [] as citation, index}
					<button
						class="underline"
						on:click={() =>
							action(async () => {
								source = (
									await api(
										`/documents/${citation.document_id}/download?revision_id=${citation.revision_id}`
									)
								).content;
							})}>[{retainedLabel(answer, index)}] {citation.title ?? 'Retained source'}</button
					>
				{/each}
				{#each answer.research?.web_sources ?? [] as citation}
					<a class="underline" href={citation.url} target="_blank" rel="noopener noreferrer"
						>[{citation.source_id}] {citation.title ?? citation.url} (not saved)</a
					>
				{/each}
			</article>
		{/each}
		<form on:submit|preventDefault={ask} class="flex gap-2">
			<input
				class="flex-1 border rounded p-2"
				aria-label="Research question"
				placeholder="Ask about your sources"
				bind:value={question}
			/>
			<button disabled={busy || !question.trim()}>Ask</button>
		</form>
	</section>
	{#if source}<pre class="whitespace-pre-wrap p-3 border rounded">{source}</pre>{/if}
</div>
