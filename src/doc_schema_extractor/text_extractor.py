"""Document text and layout extraction - pdfplumber (MIT) + openpyxl (MIT)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber
import openpyxl


@dataclass
class PageContent:
    page_num: int
    text: str
    tables: list[list[list[str | None]]] = field(default_factory=list)


@dataclass
class DocumentContent:
    path: str
    file_type: str  # "pdf" or "xlsx"
    full_text: str
    pages: list[PageContent] = field(default_factory=list)


class TextExtractor:
    """Extract text and tables from PDF and XLSX files.
    
    Libraries used:
      - pdfplumber (MIT license) for PDFs
      - openpyxl (MIT license) for XLSX
    """

    def extract(self, file_path: str | Path) -> DocumentContent:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix in (".xlsx", ".xls"):
            return self._extract_xlsx(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, path: Path) -> DocumentContent:
        pages: list[PageContent] = []
        all_text_parts: list[str] = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                tables = page.extract_tables() or []
                # Normalize table cells to strings
                normalized_tables = [
                    [[str(cell) if cell is not None else "" for cell in row] for row in table]
                    for table in tables
                ]
                pages.append(PageContent(page_num=i + 1, text=text, tables=normalized_tables))
                all_text_parts.append(text)

        return DocumentContent(
            path=str(path),
            file_type="pdf",
            full_text="\n".join(all_text_parts),
            pages=pages,
        )

    def _extract_xlsx(self, path: Path) -> DocumentContent:
        wb = openpyxl.load_workbook(path, data_only=True)
        pages: list[PageContent] = []
        all_text_parts: list[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[list[str]] = []
            text_lines: list[str] = []

            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(cells):
                    rows.append(cells)
                    text_lines.append("  ".join(cells))

            text = "\n".join(text_lines)
            pages.append(PageContent(page_num=len(pages) + 1, text=text, tables=[rows]))
            all_text_parts.append(f"[Sheet: {sheet_name}]\n{text}")

        return DocumentContent(
            path=str(path),
            file_type="xlsx",
            full_text="\n".join(all_text_parts),
            pages=pages,
        )
