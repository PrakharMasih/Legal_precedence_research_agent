from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from src.core.exceptions import ValidationError


@dataclass(slots=True)
class ChunkSlice:
    """A chunk of text with position and hierarchy metadata.

    Parents are large context blocks (~2000 chars) used for LLM reasoning.
    Children are small retrieval units (~700 chars) used for embedding/search.
    Each child has a ``parent_index`` pointing to its parent in the parent sublist
    returned by :meth:`Chunker.chunk_text`.
    """

    content: str
    char_start: int
    char_end: int
    section: str = "other"
    chunk_type: str = "child"
    parent_index: int | None = None  # index into the parent sublist


@dataclass(slots=True)
class _LegalSection:
    name: str
    content: str
    char_offset: int  # byte offset of content start within the full document


# ── Chunk size constants ──────────────────────────────────────────────────────
# Legal text ≈ 4 chars/token.  Parent ≈ 500 tokens (reasoning context for LLM).
# Child ≈ 175 tokens (precise unit for dense/sparse retrieval).
_PARENT_SIZE: int = 2000
_PARENT_OVERLAP: int = 400  # 20 %
_CHILD_SIZE: int = 700
_CHILD_OVERLAP: int = 140  # 20 %

# ── Section-header patterns ───────────────────────────────────────────────────
# Each pattern matches a standalone line (with optional leading numbering) that
# marks the start of a named section in an Indian court judgment.
_SECTION_HEADERS: list[tuple[str, re.Pattern[str]]] = [
    (
        "facts",
        re.compile(
            r"(?m)^\s*(?:\d+[\.\)]\s*|[IVX]+[\.\)]\s*)?"
            r"(?:FACTS?(?:\s+OF\s+THE\s+CASE)?|FACTUAL\s+BACKGROUND"
            r"|BACKGROUND|BRIEF\s+FACTS?|STATEMENT\s+OF\s+FACTS?|RELEVANT\s+FACTS?)"
            r"\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "issues",
        re.compile(
            r"(?m)^\s*(?:\d+[\.\)]\s*|[IVX]+[\.\)]\s*)?"
            r"(?:ISSUES?(?:\s+FOR\s+(?:CONSIDERATION|DETERMINATION))?"
            r"|QUESTION[S]?\s+OF\s+LAW"
            r"|POINTS?\s+(?:FOR\s+DETERMINATION|IN\s+DISPUTE|OF\s+LAW|RAISED)"
            r"|POINTS?\s+FOR\s+DECISION)"
            r"\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "arguments",
        re.compile(
            r"(?m)^\s*(?:\d+[\.\)]\s*|[IVX]+[\.\)]\s*)?"
            r"(?:SUBMISSIONS?(?:\s+(?:OF|BY)\s+\w+(?:\s+\w+)*)?|ARGUMENTS?|CONTENTIONS?)"
            r"\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "findings",
        re.compile(
            r"(?m)^\s*(?:\d+[\.\)]\s*|[IVX]+[\.\)]\s*)?"
            r"(?:FINDINGS?|ANALYSIS|REASONING(?:\s+AND\s+FINDINGS?)?"
            r"|DISCUSSION|OBSERVATIONS?|CONSIDERATION[S]?)"
            r"\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "judgment",
        re.compile(
            r"(?m)^\s*(?:\d+[\.\)]\s*|[IVX]+[\.\)]\s*)?"
            r"(?:JUDGMENT|DECISION|ORDER|HELD|CONCLUSION[S]?|RESULT"
            r"|DECREE|OPERATIVE\s+(?:PART|ORDER))"
            r"\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
]


class Chunker:
    """Hybrid Hierarchical Chunker for legal judgment documents.

    Step 1 – Structure-aware split: detects legal sections (Facts, Issues,
              Arguments, Findings, Judgment) using header patterns.
    Step 2 – Recursive parent chunking: each section is split into ~2000-char
              parent blocks with 20 % overlap, preserving reasoning context.
    Step 3 – Child derivation: each parent is further split into ~700-char
              child units with 20 % overlap for precise embedding / retrieval.
    Step 4 – Metadata: every chunk carries ``section`` and ``chunk_type``
              so the pipeline can attach case_name, court, date, etc.

    :meth:`chunk_text` returns **parents first** (``chunk_type="parent"``)
    followed by all children (``chunk_type="child"``).  Each child's
    ``parent_index`` references its parent's position in the parent sublist.
    """

    def __init__(
        self,
        parent_size: int = _PARENT_SIZE,
        parent_overlap: int = _PARENT_OVERLAP,
        child_size: int = _CHILD_SIZE,
        child_overlap: int = _CHILD_OVERLAP,
    ) -> None:
        self._parent_size = parent_size
        self._parent_overlap = parent_overlap
        self._child_size = child_size
        self._child_overlap = child_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    async def chunk_text(self, text: str) -> list[ChunkSlice]:
        """Return [parents…, children…] for *text*.

        Parents and children are both ``ChunkSlice`` objects distinguished by
        ``chunk_type``.  The caller (pipeline) separates them and stores /
        indexes them independently.
        """
        if not text.strip():
            return []
        return await asyncio.to_thread(self._chunk_sync, text)

    # ── Sync implementation ───────────────────────────────────────────────────

    def _chunk_sync(self, text: str) -> list[ChunkSlice]:
        sections = self._detect_sections(text)
        if not sections:
            sections = [_LegalSection("other", text.strip(), 0)]

        parent_slices: list[ChunkSlice] = []
        child_groups: list[list[ChunkSlice]] = []

        for section in sections:
            parents = self._make_parent_chunks(section, text)
            for parent in parents:
                parent_idx = len(parent_slices)
                parent_slices.append(parent)
                child_groups.append(self._make_child_chunks(parent, parent_idx, text))

        if not parent_slices:
            raise ValidationError("Chunker produced no parent chunks from non-empty text")

        all_children = [child for group in child_groups for child in group]
        return parent_slices + all_children

    # ── Section detection ─────────────────────────────────────────────────────

    def _detect_sections(self, text: str) -> list[_LegalSection]:
        """Find legal section boundaries and label them."""
        raw: list[tuple[int, int, str]] = []
        for section_name, pattern in _SECTION_HEADERS:
            for m in pattern.finditer(text):
                raw.append((m.start(), m.end(), section_name))

        if not raw:
            return []

        raw.sort(key=lambda t: t[0])

        # Remove overlapping matches (keep first)
        deduped: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, name in raw:
            if start >= last_end:
                deduped.append((start, end, name))
                last_end = end

        sections: list[_LegalSection] = []

        # Preamble: substantial text before the first header (case header, parties, etc.)
        preamble_text = text[: deduped[0][0]].strip()
        if len(preamble_text) > 150:
            sections.append(_LegalSection("preamble", preamble_text, 0))

        for i, (_start, end, name) in enumerate(deduped):
            next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
            content = text[end:next_start].strip()
            if content:
                sections.append(_LegalSection(name, content, end))

        return sections

    # ── Parent & child chunk creation ─────────────────────────────────────────

    def _make_parent_chunks(self, section: _LegalSection, full_text: str) -> list[ChunkSlice]:
        units = self._split_into_units(section.content)
        return self._merge_units(
            units=units,
            max_size=self._parent_size,
            overlap=self._parent_overlap,
            chunk_type="parent",
            section=section.name,
            full_text=full_text,
            search_hint=section.char_offset,
        )

    def _make_child_chunks(
        self, parent: ChunkSlice, parent_idx: int, full_text: str
    ) -> list[ChunkSlice]:
        units = self._split_into_units(parent.content)
        return self._merge_units(
            units=units,
            max_size=self._child_size,
            overlap=self._child_overlap,
            chunk_type="child",
            section=parent.section,
            full_text=full_text,
            search_hint=parent.char_start,
            parent_index=parent_idx,
        )

    # ── Text unit splitting ───────────────────────────────────────────────────

    def _split_into_units(self, text: str) -> list[str]:
        """Split text into paragraph-level units ≤ child_size where possible."""
        blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
        if not blocks:
            return [text.strip()] if text.strip() else []

        units: list[str] = []
        for block in blocks:
            if len(block) <= self._child_size:
                units.append(block)
            else:
                lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
                if len(lines) > 1:
                    # Merge adjacent short lines to avoid over-fragmentation
                    current = ""
                    for line in lines:
                        sep = " " if current else ""
                        if current and len(current) + 1 + len(line) > self._child_size:
                            units.append(current)
                            current = line
                        else:
                            current = current + sep + line
                    if current:
                        units.append(current)
                else:
                    # Single long line: split on sentence boundaries
                    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\(])", block)
                    if len(sentences) > 1:
                        current = ""
                        for sent in sentences:
                            sep = " " if current else ""
                            if current and len(current) + 1 + len(sent) > self._child_size:
                                units.append(current)
                                current = sent.strip()
                            else:
                                current = current + sep + sent.strip()
                        if current:
                            units.append(current)
                    else:
                        units.append(block)  # _hard_split will handle it in merge

        return units or [text.strip()]

    # ── Merging units into fixed-size chunks ──────────────────────────────────

    def _merge_units(
        self,
        units: list[str],
        max_size: int,
        overlap: int,
        chunk_type: str,
        section: str,
        full_text: str,
        search_hint: int,
        parent_index: int | None = None,
    ) -> list[ChunkSlice]:
        """Merge text units into overlapping fixed-size chunks."""
        result: list[ChunkSlice] = []
        current: list[str] = []
        current_len = 0
        cursor = search_hint

        def flush() -> None:
            nonlocal cursor
            if not current:
                return
            content = " ".join(current).strip()
            if not content:
                return
            char_start, char_end = self._locate(content, full_text, cursor)
            result.append(
                ChunkSlice(
                    content=content,
                    char_start=char_start,
                    char_end=char_end,
                    section=section,
                    chunk_type=chunk_type,
                    parent_index=parent_index,
                )
            )
            cursor = max(cursor, char_end - overlap)

        for unit in units:
            unit_len = len(unit)

            # Safety: hard-split a unit that already exceeds max_size
            if unit_len > max_size:
                if current:
                    flush()
                    current = []
                    current_len = 0
                result.extend(
                    self._hard_split(
                        unit, max_size, chunk_type, section, full_text, cursor, parent_index
                    )
                )
                if result:
                    cursor = max(cursor, result[-1].char_end)
                continue

            sep = 1 if current else 0
            if current and current_len + sep + unit_len > max_size:
                flush()
                # Retain ~overlap chars from the end of current as overlap context
                overlap_parts: list[str] = []
                accumulated = 0
                for part in reversed(current):
                    if accumulated + len(part) + 1 > overlap:
                        break
                    overlap_parts.insert(0, part)
                    accumulated += len(part) + 1
                current = overlap_parts + [unit]
                current_len = accumulated + unit_len
            else:
                current.append(unit)
                current_len += sep + unit_len

        flush()

        # Fallback: if still empty (e.g. units list was empty), hard-split joined text
        if not result and units:
            result.extend(
                self._hard_split(
                    " ".join(units),
                    max_size,
                    chunk_type,
                    section,
                    full_text,
                    search_hint,
                    parent_index,
                )
            )

        return result

    # ── Hard split (character-level fallback) ─────────────────────────────────

    def _hard_split(
        self,
        text: str,
        max_size: int,
        chunk_type: str,
        section: str,
        full_text: str,
        search_hint: int,
        parent_index: int | None,
    ) -> list[ChunkSlice]:
        """Character-level fallback split for text that exceeds max_size."""
        result: list[ChunkSlice] = []
        pos = 0
        cursor = search_hint
        while pos < len(text):
            end = min(pos + max_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", pos, end)
                if boundary > pos:
                    end = boundary
            content = text[pos:end].strip()
            if content:
                char_start, char_end = self._locate(content, full_text, cursor)
                result.append(
                    ChunkSlice(
                        content=content,
                        char_start=char_start,
                        char_end=char_end,
                        section=section,
                        chunk_type=chunk_type,
                        parent_index=parent_index,
                    )
                )
                cursor = char_end
            pos = end + 1
        return result

    # ── Position lookup ───────────────────────────────────────────────────────

    def _locate(self, content: str, full_text: str, hint: int) -> tuple[int, int]:
        """Locate content's char_start in full_text, searching near hint."""
        needle = content[:80].strip()
        if not needle:
            return hint, hint + len(content)
        search_from = max(0, hint - 100)
        pos = full_text.find(needle, search_from)
        if pos == -1:
            pos = full_text.find(needle)
        if pos == -1:
            pos = hint
        return pos, pos + len(content)
