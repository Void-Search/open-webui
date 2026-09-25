<script lang="ts">
	import { onMount } from 'svelte';
	import AskUserCard from '$lib/components/chat/AskUserCard.svelte';

	export let messageId: string;
	export let recovery: any = null;
	export let done = false;
	export let active = true;
	export let readOnly = false;
	export let submit: (reply: string) => Promise<void>;

	let show = false;
	let mounted = false;
	let offered = false;
	let wasPending = !done;
	let submitting = false;
	let error = '';
	$: key = `research-recovery:${messageId}`;
	$: questions = [
		{
			id: messageId,
			header: recovery?.kind === 'clarification' ? 'Clarification' : 'Continue research',
			question: recovery?.question ?? 'How would you like to continue?',
			options: (recovery?.choices ?? []).map((choice: { label: string; description?: string }) => ({
				label: choice.label,
				description: choice.description ?? ''
			}))
		}
	];
	onMount(() => {
		try {
			offered = localStorage.getItem(key) === 'seen';
		} catch {
			/* Session-only state. */
		}
		mounted = true;
	});
	$: if (!done) wasPending = true;
	$: if (mounted && done && wasPending && active && !readOnly && recovery && !offered) {
		offered = true;
		show = true;
		try {
			localStorage.setItem(key, 'seen');
		} catch {
			/* Session-only state. */
		}
	}
	$: if (!active || readOnly) show = false;

	async function send(reply: string) {
		if (submitting || !reply.trim() || !active || !done || readOnly) return;
		submitting = true;
		error = '';
		try {
			await submit(reply.trim());
			show = false;
		} catch {
			error = 'Could not submit your answer. Please try again.';
			submitting = false;
			show = true;
		}
	}

	function answer(event: CustomEvent) {
		const answer = event.detail?.answers?.[messageId];
		const reply =
			answer?.type === 'other' ? answer.text : recovery?.choices?.[answer?.option_index]?.reply;
		if (typeof reply === 'string') void send(reply);
	}
</script>

{#if recovery && done && active && !readOnly}
	<button
		class="mt-2 rounded-lg border px-3 py-1.5 text-sm"
		disabled={submitting}
		on:click={() => (show = true)}
	>
		Clarify or retry
	</button>
{/if}
{#if show && active && done && !readOnly && !submitting}
	<div data-testid="research-recovery">
		<AskUserCard
			bind:show
			{questions}
			recommendFirst={recovery?.kind === 'retry'}
			maxAnswerLength={1000}
			on:confirm={answer}
			on:cancel={() => (show = false)}
		/>
		{#if error}<p role="alert" class="text-sm text-red-600">{error}</p>{/if}
	</div>
{/if}
