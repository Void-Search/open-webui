import { WEBUI_API_BASE_URL } from '$lib/constants';

export type AdaptiveTask = {
	id: string;
	status: string;
	stage: string;
	skill?: string | null;
	stop_reason: string | null;
	answer: string | null;
	pending: {
		id: string;
		kind: string;
		question?: string;
		binding?: string;
		operation?: unknown;
	} | null;
	used: Record<string, number>;
	results?: Array<{ evidence?: AdaptiveEvidence[] | { evidence?: AdaptiveEvidence[] } | null }>;
	artifacts: Array<{
		id: string;
		workspace: string;
		path: string;
		sha256: string;
		download_available: boolean;
	}>;
};

export type AdaptiveCapabilities = { terminal: boolean };

export async function adaptiveCapabilities(token: string): Promise<AdaptiveCapabilities> {
	const value = await adaptiveRequest(token, 'capabilities');
	return { terminal: value?.terminal === true };
}

type AdaptiveEvidence = {
	text?: string;
	source_id?: string;
	url?: string | null;
	source?: { document_id?: string; revision_id?: string; title?: string };
	citation?: { url?: string; title?: string } | null;
};

export function adaptiveSources(task: AdaptiveTask) {
	const sources = new Map<string, any>();
	for (const result of task.results ?? []) {
		const passages = Array.isArray(result.evidence)
			? result.evidence
			: (result.evidence?.evidence ?? []);
		for (const passage of passages) {
			if (!passage.text || !passage.source_id) continue;
			const document = passage.source?.document_id;
			const revision = passage.source?.revision_id;
			const url = passage.url ?? passage.citation?.url;
			const identity = document && revision ? `research:${document}:${revision}` : url;
			if (!identity) continue;
			const link =
				document && revision
					? `/api/v1/ravenous/research/knowledge/documents/${encodeURIComponent(document)}/download?revision_id=${encodeURIComponent(revision)}`
					: url;
			const source = {
				source: {
					id: identity,
					name: passage.source?.title ?? passage.citation?.title ?? url ?? document,
					type: 'research'
				},
				document: [passage.text],
				metadata: [
					{
						...passage.source,
						source: identity,
						link,
						source_url: url,
						research_kind: document ? 'saved' : 'web'
					}
				]
			};
			const existing = sources.get(passage.source_id);
			if (existing) {
				existing.document.push(...source.document);
				existing.metadata.push(...source.metadata);
			} else sources.set(passage.source_id, source);
		}
	}
	return [...sources.entries()]
		.sort(([a], [b]) => Number(a.slice(1)) - Number(b.slice(1)))
		.map(([, source]) => source);
}

export async function downloadAdaptiveArtifact(
	token: string,
	taskId: string,
	artifact: AdaptiveTask['artifacts'][number]
) {
	const response = await fetch(
		`${WEBUI_API_BASE_URL}/ravenous/adaptive/tasks/${taskId}/artifacts/${artifact.id}`,
		{
			headers: { Authorization: `Bearer ${token}` }
		}
	);
	if (!response.ok) throw new Error('Artifact has changed or is unavailable');
	const url = URL.createObjectURL(await response.blob());
	try {
		const link = document.createElement('a');
		link.href = url;
		link.download = artifact.path.split('/').pop() ?? 'artifact';
		link.click();
	} finally {
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}
}

export async function adaptiveRequest(token: string, path: string, body?: unknown, key?: string) {
	const response = await fetch(`${WEBUI_API_BASE_URL}/ravenous/adaptive/${path}`, {
		method: body === undefined ? 'GET' : 'POST',
		headers: {
			Authorization: `Bearer ${token}`,
			'Content-Type': 'application/json',
			...(key ? { 'Idempotency-Key': key } : {})
		},
		...(body === undefined ? {} : { body: JSON.stringify(body) })
	});
	const value = await response.json();
	if (!response.ok)
		throw new Error(value?.error?.message ?? value?.detail ?? 'Task request failed');
	return value;
}

export async function watchAdaptiveTask(
	token: string,
	id: string,
	after: string,
	signal: AbortSignal,
	onEvent: (value: any, eventId: string) => void
) {
	const response = await fetch(`${WEBUI_API_BASE_URL}/ravenous/adaptive/tasks/${id}/events`, {
		headers: { Authorization: `Bearer ${token}`, 'Last-Event-ID': after || '0' },
		signal
	});
	if (!response.ok || !response.body) throw new Error('Task progress is unavailable');
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	try {
		while (true) {
			const { value, done } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			let boundary: number;
			while ((boundary = buffer.indexOf('\n\n')) >= 0) {
				const frame = buffer.slice(0, boundary);
				buffer = buffer.slice(boundary + 2);
				const lines = frame.split('\n');
				const data = lines
					.filter((line) => line.startsWith('data: '))
					.map((line) => line.slice(6))
					.join('\n');
				const eventId = lines.find((line) => line.startsWith('id: '))?.slice(4) ?? '';
				if (data) onEvent(JSON.parse(data), eventId);
			}
		}
	} finally {
		await reader.cancel().catch(() => {});
	}
}
