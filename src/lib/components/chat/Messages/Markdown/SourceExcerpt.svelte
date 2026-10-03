<script lang="ts">
	import { sourceExcerptHtml } from './sourceExcerpt';
	export let text: string;
</script>

<div class="ravenous-source-excerpt break-words font-sans not-italic" dir="auto">
	{@html sourceExcerptHtml(text)}
</div>

<style>
	.ravenous-source-excerpt :global(pre) {
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.ravenous-source-excerpt :global(table) {
		display: block;
		overflow-x: auto;
	}
</style>
