import assert from 'node:assert/strict';
import fs from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const source = fs.readFileSync(
	new URL('../../src/lib/components/workspace/Knowledge/KnowledgeBase.svelte', import.meta.url),
	'utf8'
);
const start = source.indexOf('const collectDirectoryFiles =');
const code = stripTypeScriptTypes(
	source.slice(start, source.indexOf('const buildDirectoryManifest =', start))
);

for (const scenario of ['selected', 'cancelled', 'error', 'throw', 'unreadable']) {
	test(`directory picker: ${scenario}`, async () => {
		let removed = 0;
		const errors = [];
		const input = {
			style: {},
			files: [
				{ name: 'rag.md', webkitRelativePath: 'docs/rag.md' },
				{ name: 'contract.md', webkitRelativePath: 'docs/contracts/contract.md' },
				{ name: 'secret', webkitRelativePath: 'docs/.private/secret' }
			],
			remove() { removed++; },
			click() {
				assert.equal(this.webkitdirectory, true);
				if (scenario === 'selected') this.onchange();
				else if (scenario === 'cancelled') this.oncancel();
				else if (scenario === 'error') this.onerror(new Error('read failed'));
				else if (scenario === 'unreadable') {
					Object.defineProperty(this, 'files', { get() { throw new Error('unreadable'); } });
					this.onchange();
				} else throw new Error('picker failed');
			}
		};
		const context = vm.createContext({
			document: { createElement: () => input, body: { appendChild() {} } },
			window: { showDirectoryPicker() { assert.fail('native picker must not run'); } },
			hasHiddenFolder: (path) => path.split('/').some((part) => part.startsWith('.')),
			handleUploadError: (error) => errors.push(error.message)
		});
		const result = await vm.runInContext(`${code}\ncollectDirectoryFiles()`, context);
		assert.equal(removed, 1);
		if (scenario === 'selected') {
			assert.equal(result.length, 2);
			assert.equal(result[1].path, 'docs/contracts');
		} else assert.equal(result, null);
		assert.equal(errors.length, ['error', 'throw', 'unreadable'].includes(scenario) ? 1 : 0);
	});
}