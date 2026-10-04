<script lang="ts">
	import { onMount } from 'svelte';
	let frontendUrl = '';
	let loading = true;
	onMount(async () => {
		try {
			const response = await fetch('/api/v1/ravenous/research/config', {
				headers: { Authorization: `Bearer ${localStorage.token}` }
			});
			if (response.ok) frontendUrl = (await response.json()).frontend_url ?? '';
		} catch {
			frontendUrl = '';
		} finally {
			loading = false;
		}
	});
</script>

<svelte:head><title>Research workspace</title></svelte:head>
<div class="mx-auto max-w-2xl p-8 space-y-5">
	<h1 class="text-2xl font-semibold">Your research workspace</h1>
	<p>Organize projects, inspect evidence, and connect your ideas with editable knowledge graphs and mind maps.</p>
	<p>The workspace opens in a separate tab and uses your personal Ravenous access key. Sign in there to manage your private research.</p>
	{#if loading}
		<p role="status">Loading workspace configuration…</p>
	{:else if frontendUrl}
		<a class="inline-block rounded-xl bg-black text-white dark:bg-white dark:text-black px-5 py-3" href={frontendUrl} target="_blank" rel="noopener noreferrer">Open research workspace ↗</a>
	{:else}
		<p role="status">The research frontend has not been configured. Set connections.research_ui_public_url in the stack host configuration, or enable the workspace on the research service.</p>
	{/if}
</div>
