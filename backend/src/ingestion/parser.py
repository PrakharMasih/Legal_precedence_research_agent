from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from src.core.exceptions import IngestionError


@dataclass(slots=True)
class ParsedDocument:
    file_name: str
    file_hash: str
    raw_text: str
    page_count: int
    char_count: int
    case_name: str | None
    court_name: str | None
    judgment_date: str | None


CASE_NAME_PATTERN = re.compile(r"^(.+?\b(?:v\.?|vs\.?|versus)\b.+)$", re.IGNORECASE | re.MULTILINE)
COURT_PATTERN = re.compile(r"((?:SUPREME|HIGH) COURT[^\n]*)", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)

# ── Noise patterns stripped from every page before indexing ──────────────────
# Each pattern is applied line-by-line (after per-line strip).
_NOISE_LINE_PATTERNS: list[re.Pattern[str]] = [
    # Indian Kanoon watermark: "Indian Kanoon - http://indiankanoon.org/doc/<any number>/"
    re.compile(r"^Indian\s+Kanoon\s*[-–]\s*https?://\S+", re.IGNORECASE),
    # Manupatra / SCC Online / Westlaw / LexisNexis watermarks
    re.compile(r"^(?:www\.)?(?:manupatra|scconline|westlaw|lexisnexis)\S*$", re.IGNORECASE),
    # Standalone page numbers (1–4 digits, possibly preceded by "Page" or "Pg")
    re.compile(r"^(?:page\s*)?\d{1,4}$", re.IGNORECASE),
    # Section-break lines: "---", "===", "***" (3+ chars)
    re.compile(r"^[-=*]{3,}$"),
    # "Printed from …" or "Downloaded from …" lines
    re.compile(r"^(?:printed|downloaded|generated)\s+(?:from|by|on)\b", re.IGNORECASE),
]


def _clean_text(raw: str) -> str:
    """Remove watermarks, page numbers, and other noise from extracted PDF text."""
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if any(pat.match(stripped) for pat in _NOISE_LINE_PATTERNS):
            continue
        cleaned_lines.append(stripped)

    # Collapse runs of 3+ consecutive blank lines → 2 blank lines (paragraph break)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def parse_pdf(file_path: Path) -> ParsedDocument:
    return await asyncio.to_thread(_parse_pdf_sync, file_path)


def _parse_pdf_sync(file_path: Path) -> ParsedDocument:
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with pdfplumber.open(file_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]

    raw_text = "\n\n".join(page.strip() for page in pages if page.strip())
    if len(raw_text.strip()) <= 50:
        raise IngestionError("No readable text found")

    raw_text = _clean_text(raw_text)

    excerpt = raw_text[:2000]
    case_name_match = CASE_NAME_PATTERN.search(excerpt)
    court_match = COURT_PATTERN.search(excerpt)
    date_match = DATE_PATTERN.search(excerpt)

    return ParsedDocument(
        file_name=file_path.name,
        file_hash=file_hash,
        raw_text=raw_text,
        page_count=len(pages),
        char_count=len(raw_text),
        case_name=case_name_match.group(1).strip() if case_name_match else None,
        court_name=court_match.group(1).strip() if court_match else None,
        judgment_date=date_match.group(1).strip() if date_match else None,
    )
