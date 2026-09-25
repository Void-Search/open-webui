// Run in the isolated browser fixture: await import('/acceptance.ts').then(m => m.exercise()).
const settle = () => new Promise((resolve) => setTimeout(resolve, 200));
const button = (text: string) => {
	const value = [...document.querySelectorAll('button')].find(
		(node) => node.textContent?.trim() === text
	);
	if (!value) throw new Error(`Missing button: ${text}`);
	return value;
};
const assert = (condition: unknown, message: string) => {
	if (!condition) throw new Error(message);
};
const card = () => document.querySelector('[data-testid=research-recovery]');
const replies = () =>
	JSON.parse(document.querySelector('[data-testid=submitted]')?.textContent || '[]');

export async function exercise() {
	button('Start search').click();
	await settle();
	for (let index = 1; index <= 5; index++) {
		assert(
			document.body.innerText.includes(`Complementary search ${index}`),
			'Exact live query missing'
		);
	}
	button('Finish research').click();
	await settle();
	assert(card(), 'Recovery did not appear automatically');
	assert(!document.querySelector('[role=dialog]'), 'Clarification created a modal');
	button('Cancel').click();
	await settle();
	assert(!card(), 'Dismissal failed');
	button('Clarify or retry').click();
	await settle();
	assert(card(), 'Reopening failed');
	button('Current edition').click();
	button('Current edition').click();
	await settle();
	assert(
		replies().length === 1 && replies()[0] === 'Current edition',
		'Duplicate or incorrect submission'
	);
	assert(!card(), 'Successful submission did not close recovery');

	button('Start search').click();
	await settle();
	button('Finish research').click();
	await settle();
	button('Switch branch').click();
	await settle();
	assert(!card(), 'Card crossed the active branch');
	button('Switch branch').click();
	await settle();
	assert(!card(), 'Returning to the branch reopened a dismissed card');
	button('Clarify or retry').click();
	await settle();
	const field = document.querySelector<HTMLInputElement>('[data-testid=research-recovery] input')!;
	field.value = 'Use the current supported edition';
	field.dispatchEvent(new Event('input', { bubbles: true }));
	await settle();
	button('Submit answers').click();
	await settle();
	assert(replies()[0] === field.value, 'Custom answer was lost');
	document.querySelectorAll('details').forEach((node) => (node.open = true));
	for (const text of [
		'Website returned HTTP 403',
		'Discovery snippet',
		'Local knowledge: Completed',
		'Selected · Cited',
		'Retrieved · Excluded',
		'Duplicate content'
	]) {
		assert(document.body.innerText.includes(text), `Missing detail: ${text}`);
	}
	assert(
		document.querySelector('a[href="https://example.org/2"]'),
		'Unreadable source link was hidden'
	);
	return {
		checks: [
			'live queries',
			'failed pages',
			'selection and citations',
			'automatic card',
			'dismiss and reopen',
			'duplicate prevention',
			'branch isolation',
			'custom answer'
		]
	};
}

export async function reloaded() {
	assert(!card(), 'Completed card reopened automatically after reload');
	assert(
		document.body.innerText.includes('Research complete'),
		'Research state did not survive reload'
	);
	button('Clarify or retry').click();
	await settle();
	assert(card(), 'Reloaded clarification cannot be reopened');
	return { checks: ['reload persistence', 'reopen after reload'] };
}
