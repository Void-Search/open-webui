<script lang="ts">
	import { onDestroy, onMount, createEventDispatcher } from 'svelte';
	import {
		adaptiveRequest,
		adaptiveCapabilities,
		adaptiveSources,
		downloadAdaptiveArtifact,
		watchAdaptiveTask,
		type AdaptiveTask
	} from '$lib/apis/adaptive';
	import Citations from './Messages/Citations.svelte';
	export let taskId = '';
	export let temporary = false;
	export let helper = false;
	export let initialPrompt = '';
	export let contextMessages: Array<{ role: string; content: string }> = [];
	export let sessionId = '';
	const dispatch = createEventDispatcher();
	let prompt = initialPrompt;
	let depth = 'auto';
	let task: AdaptiveTask | null = null;
	let answer = '';
	let error = '';
	let busy = false;
	let controller: AbortController | null = null;
	let after = '0';
	let activeId = '';
	let completedId = '';
	let creationKey = '';
	let creationFingerprint = '';
	let documents: Array<{ id: string; title?: string }> = [];
	let selectedDocuments: string[] = [];
	let executionAvailable = false;
	let executionAllowed = false;
	let terminalAvailable = false;
	let terminalAllowed = false;
	$: terminalSelected = terminalAllowed || task?.skill === 'terminal_workspace';
	$: if (temporary || helper || !terminalAvailable) terminalAllowed = false;
	$: if (terminalAllowed) {
		executionAllowed = false;
		selectedDocuments = [];
		documents = [];
	}
	let expanded = Boolean(taskId);
	const stopped = new Set([
		'completed',
		'failed',
		'cancelled',
		'interrupted',
		'waiting_user',
		'waiting_approval'
	]);
	onMount(async () => {
		await Promise.allSettled([
			adaptiveRequest(localStorage.token, 'config').then((config) => {
				executionAvailable = config.execution === true;
			}),
			adaptiveCapabilities(localStorage.token).then((capabilities) => {
				terminalAvailable = capabilities.terminal;
			})
		]);
	});

	async function loadSources() {
		try {
			const result = await adaptiveRequest(localStorage.token, 'sources');
			documents = (Array.isArray(result) ? result : (result.data ?? result.items ?? [])).map(
				(document: any) => ({ id: document.document_id ?? document.id, title: document.title })
			);
		} catch (failure) {
			error = String(failure);
		}
	}
	$: if (taskId && taskId !== activeId) attach(taskId);
	$: if (taskId) expanded = true;
	$: if (!taskId && activeId) {
		controller?.abort();
		activeId = '';
		task = null;
		after = '0';
	}

	async function attach(id: string) {
		controller?.abort();
		activeId = id;
		after = '0';
		controller = new AbortController();
		const current = controller;
		try {
			const initial = await adaptiveRequest(localStorage.token, `tasks/${id}`);
			if (current.signal.aborted || activeId !== id) return;
			task = initial;
			while (!current.signal.aborted && task && !stopped.has(task.status)) {
				try {
					await watchAdaptiveTask(
						localStorage.token,
						id,
						after,
						current.signal,
						(event, sequence) => {
							if (current.signal.aborted || activeId !== id) return;
							if (sequence) after = sequence;
							// Event IDs are replay ordinals; only the attached task owns its ID.
							if (event.answer !== undefined) task = { ...event, id };
							else if (task) task = { ...task, ...event, id };
						}
					);
				} catch (failure) {
					if (current.signal.aborted) return;
					await new Promise((resolve) => setTimeout(resolve, 1000));
				}
				if (!current.signal.aborted) {
					const updated = await adaptiveRequest(localStorage.token, `tasks/${id}`);
					if (current.signal.aborted || activeId !== id) return;
					task = updated;
				}
			}
			if (
				!current.signal.aborted &&
				task?.answer &&
				completedId !== id &&
				task.status === 'completed'
			) {
				completedId = id;
				dispatch('complete', { task });
			}
		} catch (failure) {
			if (!current.signal.aborted) error = String(failure);
		}
	}

	async function start() {
		busy = true;
		error = '';
		try {
			const taskPrompt = prompt;
			const useTerminal = terminalAllowed && terminalAvailable && !temporary && !helper;
			const body = {
				chat: {
					messages: [...contextMessages.slice(-7), { role: 'user', content: taskPrompt }],
					...(!useTerminal && selectedDocuments.length
						? { knowledge: { document_ids: selectedDocuments } }
						: {}),
					...(sessionId
						? { context: { session_id: sessionId, temporary, memory_enabled: false } }
						: {})
				},
				depth,
				temporary,
				...(useTerminal
					? { skill: 'terminal_workspace', tools: ['terminal'] }
					: {
							tools: [
								'file_search',
								'web_search',
								'helpers',
								...(executionAllowed ? ['sandbox'] : [])
							]
						})
			};
			const fingerprint = JSON.stringify(body);
			if (fingerprint !== creationFingerprint) {
				creationFingerprint = fingerprint;
				creationKey = crypto.randomUUID();
			}
			task = await adaptiveRequest(localStorage.token, 'tasks', body, creationKey);
			taskId = task!.id;
			dispatch('started', { taskId, prompt: taskPrompt });
		} catch (failure) {
			error = String(failure);
		} finally {
			busy = false;
		}
	}

	function newTask() {
		controller?.abort();
		task = null;
		taskId = '';
		activeId = '';
		creationKey = '';
		creationFingerprint = '';
		terminalAllowed = false;
		executionAllowed = false;
		selectedDocuments = [];
		prompt = '';
		error = '';
		dispatch('reset');
	}

	async function download(artifact: AdaptiveTask['artifacts'][number]) {
		if (!task) return;
		try {
			await downloadAdaptiveArtifact(localStorage.token, task.id, artifact);
		} catch (failure) {
			error = String(failure);
		}
	}

	async function operate(operation: string, body: unknown = {}) {
		if (!task) return;
		busy = true;
		error = '';
		try {
			task = await adaptiveRequest(localStorage.token, `tasks/${task.id}/${operation}`, body);
			answer = '';
			busy = false;
			await attach(task!.id);
		} catch (failure) {
			error = String(failure);
		} finally {
			busy = false;
		}
	}

	onDestroy(() => controller?.abort());
</script>

<details
	bind:open={expanded}
	class="mx-auto w-full max-w-3xl max-h-[50vh] overflow-y-auto shrink-0 rounded-xl border border-gray-200 dark:border-gray-800 p-3 mt-14 mb-2"
>
	<summary class="cursor-pointer text-sm font-medium">Adaptive task</summary>
	<div class="mt-3 space-y-3 text-sm">
		{#if !task}
			<p>
				Run a task with automatic tool selection and answer checks. Progress continues if this page
				closes.
			</p>
			<textarea
				aria-label="Adaptive task prompt"
				bind:value={prompt}
				class="w-full rounded-lg border dark:border-gray-700 bg-transparent p-2"
				rows="3"
				placeholder="What should the task accomplish?"
			></textarea>
			<div class="flex items-center gap-3">
				<select
					aria-label="Task depth"
					bind:value={depth}
					class="rounded-lg bg-gray-100 dark:bg-gray-800 p-2"
				>
					<option value="auto">Auto</option><option value="fast">Fast</option><option
						value="thorough">Thorough</option
					>
				</select>
				<button
					disabled={busy || !prompt.trim()}
					on:click={start}
					class="rounded-lg bg-gray-900 text-white dark:bg-white dark:text-gray-900 px-3 py-2 disabled:opacity-50"
					>Run task</button
				>
			</div>
		{:else}
			<p aria-live="polite">
				{task.status.replaceAll('_', ' ')} · {task.stage} · {task.used.tools ?? 0} tool calls
			</p>
			{#if task.pending?.kind === 'question'}
				<p>{task.pending.question}</p>
				<input
					aria-label="Clarification answer"
					bind:value={answer}
					class="w-full rounded-lg border dark:border-gray-700 bg-transparent p-2"
				/>
				<button
					disabled={busy || !answer.trim()}
					on:click={() =>
						operate('resolve', {
							action_id: task!.pending!.id,
							answer,
							...(!terminalSelected && selectedDocuments.length
								? { knowledge: { document_ids: selectedDocuments } }
								: {})
						})}>Continue</button
				>
			{/if}
			{#if task.pending?.kind === 'approval'}
				<pre class="overflow-auto whitespace-pre-wrap">{JSON.stringify(
						task.pending.operation,
						null,
						2
					)}</pre>
				<button
					disabled={busy}
					on:click={() =>
						operate('resolve', {
							action_id: task!.pending!.id,
							binding: task!.pending!.binding,
							approve: true
						})}>Approve this action</button
				>
				<button
					disabled={busy}
					on:click={() =>
						operate('resolve', {
							action_id: task!.pending!.id,
							binding: task!.pending!.binding,
							approve: false
						})}>Reject</button
				>
			{/if}
			{#if task.status === 'interrupted'}
				<p>The task was interrupted. Resume rechecks access; uncertain commands cannot replay.</p>
				<button disabled={busy} on:click={() => operate('resume')}>Resume</button>
			{/if}
			{#if !['completed', 'failed', 'cancelled'].includes(task.status)}
				<button disabled={busy} on:click={() => operate('cancel')}>Cancel task</button>
			{/if}
			{#if task.answer}<p class="whitespace-pre-wrap">{task.answer}</p>{/if}
			{#if task.results?.length}<Citations sources={adaptiveSources(task)} />{/if}
			{#if task.stop_reason}<p class="text-gray-500">
					{task.stop_reason.replaceAll('_', ' ')}
				</p>{/if}
			{#each task.artifacts ?? [] as artifact}
				<p>
					{artifact.path} <span class="text-gray-500">{artifact.sha256.slice(0, 12)}</span>
					{#if artifact.download_available}<button on:click={() => download(artifact)}
							>Download</button
						>{/if}
				</p>
			{/each}
			{#if ['completed', 'failed', 'cancelled'].includes(task.status)}
				<button on:click={newTask}>New task</button>
			{/if}
		{/if}
		{#if error}<p role="alert" class="text-red-600">{error}</p>{/if}
		{#if (!task || task.status === 'waiting_user') && !terminalSelected}
			<button on:click={loadSources}>Select knowledge files</button>
			{#each documents as document}
				<label class="flex items-center gap-2"
					><input
						type="checkbox"
						value={document.id}
						bind:group={selectedDocuments}
					/>{document.title ?? 'Knowledge file'}</label
				>
			{/each}
		{/if}
		{#if terminalAvailable && !temporary && !helper && !task}
			<label class="flex items-center gap-2">
				<input type="checkbox" bind:checked={terminalAllowed} />Persistent terminal
			</label>
		{/if}
		{#if executionAvailable && !task && !terminalAllowed}
			<label class="flex items-center gap-2"
				><input type="checkbox" bind:checked={executionAllowed} />Allow work in the isolated task
				workspace</label
			>
		{/if}
	</div>
</details>
