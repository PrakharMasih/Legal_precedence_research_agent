# Chunking Strategy — Legal Judgment RAG System

## Overview

The system uses a **Hybrid Hierarchical Chunking** strategy purpose-built for Indian court judgments. It combines three established techniques — **structure-aware splitting**, **recursive size-bounded chunking**, and **parent-child hierarchy** — to solve the core tension in legal RAG:

> *Small chunks = precise retrieval. Large chunks = sufficient reasoning context for the LLM.*

The strategy gives you both.

---

## The Problem With Naive Chunking

| Naive Approach | Why It Fails on Legal Docs |
|---|---|
| Fixed-size (e.g. every 400 chars) | Cuts across legal arguments mid-sentence; an Issue may be split from its Finding |
| Pure semantic (embedding-based) | Very expensive; unstable on dense, citation-heavy text; over-fragments |
| Single flat chunks for LLM | Too large for precise retrieval; irrelevant content fills the context window |
| No hierarchy | LLM gets a precise snippet but lacks the surrounding reasoning block |

Legal judgments have **internal logical structure**. The same argument — issue framing → evidence → reasoning → order — must stay coherent. The chunker respects this.

---

## Four-Step Workflow

```
Raw PDF Text
     │
     ▼
Step 1 ── Structure-aware section detection
     │       (Facts / Issues / Arguments / Findings / Judgment / Preamble)
     │
     ▼
Step 2 ── Recursive parent chunking per section
     │       (~2000 chars, 20% overlap)
     │
     ▼
Step 3 ── Child derivation from each parent
     │       (~700 chars, 20% overlap)
     │
     ▼
Step 4 ── Metadata attachment
             (section label, chunk_type, parent_index, char positions)
```

---

### Step 1 — Structure-Aware Section Detection

**File:** `_detect_sections()`, `_SECTION_HEADERS`

The chunker scans the raw document text for known legal section headers using a battery of regex patterns. Each pattern is written to handle real-world variation in Indian court documents:

- Optional roman numeral or numeric prefix (`I.`, `2.`, `IV)`)
- Case-insensitive matching
- Alternate phrasing (`BRIEF FACTS`, `FACTUAL BACKGROUND`, `STATEMENT OF FACTS` all map to `facts`)
- Optional trailing colon or dash

**Detected sections:**

| Section Label | Matched Headers |
|---|---|
| `preamble` | Text before the first header (case title, court, parties, citation) |
| `facts` | FACTS, FACTS OF THE CASE, BRIEF FACTS, FACTUAL BACKGROUND, RELEVANT FACTS, STATEMENT OF FACTS |
| `issues` | ISSUES, ISSUES FOR CONSIDERATION, POINTS FOR DETERMINATION, QUESTION OF LAW, POINTS RAISED |
| `arguments` | SUBMISSIONS, ARGUMENTS, CONTENTIONS, SUBMISSIONS BY / OF … |
| `findings` | FINDINGS, ANALYSIS, REASONING AND FINDINGS, DISCUSSION, OBSERVATIONS, CONSIDERATIONS |
| `judgment` | JUDGMENT, DECISION, ORDER, HELD, CONCLUSIONS, RESULT, DECREE, OPERATIVE PART |

If no headers are found (e.g. a bare text document), the entire text is treated as one `other` section and processing continues normally.

**Why this matters:** A chunk labelled `section = "findings"` carries fundamentally different legal weight than one labelled `section = "facts"`. This label travels through the pipeline into the SQLite store and QdrantDB metadata, enabling future section-filtered retrieval.

---

### Step 2 — Recursive Parent Chunking

**File:** `_make_parent_chunks()`, `_split_into_units()`, `_merge_units()`

Each detected section is independently chunked into **parent blocks**.

**Parameters:**
- Target size: **2000 characters** (~500 tokens at 4 chars/token)
- Overlap: **400 characters** (20%)

**How it works:**

1. The section text is first broken into **natural units** — paragraphs (double-newline splits), then lines, then sentences — from largest to smallest granularity.
2. Units are accumulated greedily until the next unit would exceed `parent_size`.
3. At that point the accumulated buffer is flushed as a parent chunk.
4. The overlap window (last ~400 chars) is carried forward into the next chunk so cross-boundary context is not lost.
5. Any unit that is itself larger than `parent_size` is passed to `_hard_split()` (character-level word-boundary split) as a safety valve.

**Result:** Parent chunks are **large, coherent reasoning blocks**. They are stored in SQLite (`chunk_type = "parent"`) but are **never embedded or added to the vector store**. Their purpose is purely to feed the LLM with reasoning context once a child has been retrieved.

---

### Step 3 — Child Chunk Derivation

**File:** `_make_child_chunks()`, `_merge_units()`

Each parent chunk is independently re-chunked into **child blocks** using the same `_merge_units()` logic with tighter parameters:

**Parameters:**
- Target size: **700 characters** (~175 tokens)
- Overlap: **140 characters** (20%)

Every child knows which parent it came from via `parent_index` (an integer index into the parent sublist).

**Result:** Child chunks are **small, precise retrieval units**. They are:
- Embedded by the `Embedder` (HuggingFace `all-MiniLM-L6-v2`)
- Stored in SQLite with `parent_id` pointing to the parent row
- Added to QdrantDB with metadata (`document_id`, `file_name`, `section`, `chunk_type`, `parent_id`, `char_start`, `char_end`)
- Indexed in SQLite FTS5 for sparse keyword search

---

### Step 4 — Metadata Attachment

Every `ChunkSlice` produced by the chunker carries:

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The text of this chunk |
| `char_start` | `int` | Absolute character offset in the original document |
| `char_end` | `int` | Absolute character end offset |
| `section` | `str` | Legal section label (`facts`, `issues`, `findings`, `judgment`, `preamble`, `other`) |
| `chunk_type` | `str` | `"parent"` or `"child"` |
| `parent_index` | `int \| None` | Index of this child's parent in the parent sublist (`None` for parents) |

These fields are persisted to SQLite (`chunks` table) and to QdrantDB vector metadata. The pipeline additionally attaches document-level metadata (`case_name`, `court_name`, `judgment_date`, `file_name`) at index time.

---

## Output Layout

`chunk_text()` returns a single flat list, always ordered **parents first, children second**:

```
[ parent_0, parent_1, parent_2, …, child_0, child_1, child_2, … ]
```

The pipeline uses this ordering to build a `parent_id → UUID` map before constructing child rows, so every child's `parent_id` foreign key resolves correctly.

---

## How Retrieval Uses the Hierarchy

```
Query
  │
  ▼
Hybrid retrieval (dense QdrantDB + sparse FTS5, RRF-fused)
  │     ← only child chunks are indexed here
  ▼
Top-N child chunks returned
  │
  ▼
Parent context expansion (SQL JOIN: child.parent_id → parent.content)
  │
  ▼
RankedChunk { content (child, precise), parent_content (parent, reasoning) }
  │
  ▼
LLM prompt receives:
  • child content  → "what was found"
  • parent content → "the full reasoning block around it"
```

The LLM never sees a decontextualised fragment. It always has the small snippet (for grounding) alongside the large parent block (for legal reasoning).

---

## Size Constants Rationale

```
Legal text ≈ 4 characters per token (English legal prose, citations, numbers)

Parent: 2000 chars ÷ 4 = ~500 tokens   → fits in most LLM context alongside system prompt
Child:   700 chars ÷ 4 = ~175 tokens   → well within embedding model limits; precise enough
                                           for high-recall dense + sparse retrieval
Overlap: 20% for both levels           → standard for legal text; prevents boundary blindness
                                           without excessive repetition
```

---

## Strategies Used and Their Benefits

### 1. Structure-Aware Splitting
**Benefit:** Legal meaning is preserved. A Finding chunk will never be mixed with a Facts chunk. Section labels enable future metadata-filtered queries (e.g. "retrieve only from `findings` sections").

### 2. Recursive Paragraph/Line/Sentence Decomposition
**Benefit:** Natural text boundaries are respected at every level. The system prefers paragraph splits → line splits → sentence splits → hard word-boundary splits, in that order, ensuring the least disruptive cut is always chosen.

### 3. Parent-Child Hierarchy
**Benefit:** Solves the fundamental RAG trade-off.
- Children → high retrieval precision (small, focused, fast to rank)
- Parents → high LLM reasoning quality (wide context, preserves legal argument flow)

### 4. 20% Sliding Overlap
**Benefit:** Cross-boundary arguments are not lost. A legal paragraph that spans two chunks appears in both, so a query matching either half retrieves the relevant block.

### 5. Hard-Split Safety Valve
**Benefit:** No chunk ever exceeds its configured maximum size regardless of document formatting. Long unnewlined paragraphs (common in OCR-extracted PDFs) are handled gracefully without errors.

### 6. Fully Async
**Benefit:** `chunk_text()` is `async` and offloads CPU-bound work to a thread pool via `asyncio.to_thread()`. The FastAPI server is never blocked during ingestion of large documents.

---

## What Is NOT Done (and Why)

| Excluded Technique | Reason |
|---|---|
| Pure semantic / embedding-based chunking | Requires an embedding call per candidate split — prohibitively expensive at ingestion; also unstable on citation-dense legal text |
| spaCy sentence tokenisation (previous approach) | Adds a heavy NLP dependency for marginal benefit; sentence boundaries are already handled by regex split on `.!?` followed by capital |
| Fixed-size character chunking | Breaks legal arguments at arbitrary points; no awareness of paragraph or section structure |
| Overlapping semantic windows | Produces excessive near-duplicate embeddings that pollute retrieval ranking |
