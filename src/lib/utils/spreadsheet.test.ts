import { describe, expect, it } from 'vitest';
import * as XLSX from 'xlsx';

describe('spreadsheet preview compatibility', () => {
	for (const bookType of ['xlsx', 'xls', 'ods'] as const) {
		it(`reads ${bookType} values used by previews and document extraction`, () => {
			const rows = [
				['Name', 'Count', 'Ready'],
				['Café', 7, true]
			];
			const workbook = XLSX.utils.book_new();
			XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet(rows), 'Acceptance');
			const bytes = XLSX.write(workbook, { type: 'array', bookType });
			const parsed = XLSX.read(bytes, { type: 'array' });
			expect(parsed.SheetNames).toEqual(['Acceptance']);
			expect(XLSX.utils.sheet_to_json(parsed.Sheets.Acceptance, { header: 1 })).toEqual(rows);
		});
	}
});
