# Research: Legal Precedent Research Agent

**Branch**: `001-legal-precedent-research` | **Phase**: 0 | **Date**: 2026-04-22
**Plan**: [plan.md](plan.md)

---

## RQ-001: QdrantDB Hybrid Search Capabilities

**Question**: Does QdrantDB natively support sparse/BM25 search, or must that be supplied externally?

**Decision**: Use SQLite FTS5 for BM25-ranked sparse retrieval alongside QdrantDB for dense retrieval. Merge with Reciprocal Rank Fusion (RRF).

**Rationale**: QdrantDB (as of v0.5) supports only dense vector search. It does not expose BM25 or TF-IDF natively. Because SQLite is already in the stack for metadata, FTS5 (enabled by default in CPython's `sqlite3`) provides BM25 scoring at zero additional dependency cost via the `bm25()` auxiliary function available in `ORDER BY rank` queries. This keeps the hybrid stack entirely embedded with no new infrastructure.

**Alternatives considered**:
- `rank_bm25` in-memory library: simpler but requires loading all chunk text into RAM; does not scale beyond ~10k chunks without memory pressure.
- Elasticsearch / OpenSearch: eliminates the constraint of single-server embedded stack; out of scope for v1.
- QdrantDB with Qdrant's `where_document` filter: text filtering only, not ranked BM25 scoring; insufficient for proper sparse retrieval.

**Key implementation detail**: SQLite FTS5 with `content='chunks'` (external content table) syncs the FTS index from the `chunks` table. BM25 rank is accessed via `ORDER BY rank` on FTS5 queries (SQLite's built-in BM25 implementation).

---

## RQ-002: LangChain + SpaCy Chunking Strategy

**Question**: What is the correct way to integrate LangChain's SpaCyTextSplitter with the specified chunk parameters (200–500 chars, 50–100 char overlap, sentence-boundary splits)?

**Decision**: Use `langchain_text_splitters.SpacyTextSplitter` with `chunk_size=400`, `chunk_overlap=75`, `pipeline="en_core_web_sm"`, `length_function=len`.

**Rationale**:
- `SpacyTextSplitter` uses spaCy's sentence segmentation (`sents` component) to identify sentence boundaries, then merges sentences into chunks up to `chunk_size` characters. This guarantees no mid-sentence or mid-word breaks.
- `chunk_size=400`: midpoint of the 200–500 range, leaving headroom for sentences that push toward the upper bound.
- `chunk_overlap=75`: midpoint of the 50–100 range. Preserves cross-sentence context at chunk boundaries.
- `en_core_web_sm`: efficient English model (12 MB). Legal text is formal English; the `sm` model is sufficient for sentence segmentation. `en_core_web_lg` would add word vectors but is unnecessary since we use separate embeddings.
- `length_function=len`: measures by character count, directly matching the spec's 200–500 char constraint.

**Alternatives considered**:
- `RecursiveCharacterTextSplitter`: splits on `\n\n`, `\n`, ` ` — does not guarantee sentence boundaries; can break mid-sentence.
- `NLTKTextSplitter`: sentence-aware but NLTK's `punkt` tokeniser is less robust on legal citation patterns (e.g., `para. 12`, `S. 123`) than spaCy's rule-based segmenter.
- Custom regex splitter: fragile against varied PDF text extraction artifacts.

**Key implementation detail**: spaCy model must be downloaded at setup time (`python -m spacy download en_core_web_sm`). The chunker runs synchronously; it should be called from a thread pool executor in async context to avoid blocking the event loop.

---

## RQ-003: Dynamic Agent Architecture (FR-010)

**Question**: How do we implement an agent that dynamically determines its own workflow without hardcoded if-else routing?

**Decision**: Use an LLM tool-calling agent (OpenAI-compatible function-calling API) implemented via LangChain's `create_tool_calling_agent` + `AgentExecutor`. The agent is given a set of named tools and a system prompt; the LLM decides which tools to call and in what sequence.

**Rationale**: Tool-calling (function-calling) is the standard pattern for dynamic agent routing. The LLM reasons over the query and selects tools autonomously. For a deep research query it will call `search_corpus` multiple times with different query angles, then `build_analysis_section` for each of supporting/adverse/strategy. For a simple corpus query it may call `search_corpus` once and return. No application-level branching is needed.

**Agent tools defined**:
| Tool | Purpose |
|------|---------|
| `search_corpus(query, n_results, search_mode)` | Hybrid retrieval — returns ranked chunks with source refs |
| `get_document_summary(document_id)` | Fetch full text + metadata for a specific judgment |
| `build_precedent_entry(document_id, role, reasoning)` | Construct a supporting or adverse precedent entry |
| `finalize_research_response(supporting, adverse, strategy)` | Assemble the structured `PrecedentAnalysis` output |
| `finalize_general_response(answer, documents)` | Assemble a `GeneralQueryResponse` output |

**System prompt strategy**: The prompt tells the agent its role (legal research associate), describes when and how to use each tool, and instructs it to cite only documents it has actually retrieved. It does not prescribe a fixed sequence of calls.

**Alternatives considered**:
- LangGraph stateful graph: more control, but introduces explicit state machine edges — effectively hardcoded routing. Violates FR-010.
- ReAct (Reason + Act) prompting without function calling: text-only, harder to parse structured outputs reliably.
- Hardcoded pipeline with LLM at each step: directly violates FR-010.

**Key implementation detail**: The AgentExecutor `max_iterations` should be capped (e.g., 15) to prevent runaway tool loops. Tool errors are surfaced back to the agent as tool return values so it can recover gracefully.

---

## RQ-004: Reciprocal Rank Fusion (RRF)

**Question**: What fusion algorithm merges dense and sparse ranked lists, and what are the key parameters?

**Decision**: Reciprocal Rank Fusion (RRF) with `k=60`.

**Formula**: For each chunk retrieved from either list, its RRF score is:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $R$ is the set of ranked lists (dense + sparse), $\text{rank}_r(d)$ is the 1-based rank of document $d$ in list $r$, and $k=60$ is a smoothing constant that reduces the impact of very high ranks.

**Rationale**: RRF is score-independent (no normalisation needed across different score scales), simple to implement, and consistently outperforms score-based fusion in retrieval benchmarks (Cormack et al., 2009). It requires no training and handles the different score distributions from QdrantDB (cosine similarity, 0–1) and SQLite FTS5 (BM25, unbounded negative) without normalisation.

**Implementation**:
1. Run dense search: top-20 chunks from QdrantDB → `dense_results = [(chunk_id, rank)]`
2. Run sparse search: top-20 chunks from SQLite FTS5 → `sparse_results = [(chunk_id, rank)]`
3. For each unique chunk in the union: compute `1/(60+rank_dense) + 1/(60+rank_sparse)` (missing rank → treated as rank=∞, contributing 0)
4. Sort by descending RRF score, return top-k (configurable, default 10)

**Alternatives considered**:
- Weighted score fusion: requires normalising incompatible score scales; sensitive to outliers.
- CombSUM / CombMNZ: better documented but score-dependent.
- Cross-encoder re-ranking after RRF: adds latency (~200ms) but improves precision; deferred to v2.

---

## RQ-005: sentence-transformers all-MiniLM-L6-v2

**Question**: Is `all-MiniLM-L6-v2` appropriate for Indian court judgment text? What are the key operational characteristics?

**Decision**: Use `all-MiniLM-L6-v2`. It is appropriate for v1.

**Characteristics**:
- Output dimension: **384**
- Max input tokens: **256** (≈1,500 chars). The chunk ceiling of 500 chars stays well within this.
- Model size: ~23 MB (quantised) — 80 MB (full). Ships with sentence-transformers; no separate download.
- Throughput: ~2,000–5,000 sentences/sec on CPU (single thread). A 50-doc corpus of ~4,000–5,000 chunks embeds in under 5 seconds on a modern CPU.
- Semantic quality: Trained on 1B+ sentence pairs. Captures general semantic similarity. Indian legal English is formal and consistent; domain-specific fine-tuning is not required for v1.

**Key implementation detail**: `SentenceTransformer.encode()` is synchronous and CPU-bound. It must run in a `ThreadPoolExecutor` to avoid blocking the async event loop. Batch encode at ingestion time (batch_size=64 optimal for CPU). At query time, encode the single query string — negligible latency (~5ms).

**Alternatives considered**:
- `all-mpnet-base-v2` (768-dim): higher quality but 4× slower on CPU; unnecessary for v1.
- `legal-bert-base-uncased`: domain-specific; better for legal classification tasks but similar retrieval performance to MiniLM on semantic similarity benchmarks.
- OpenAI `text-embedding-3-small`: excellent quality but requires API key and introduces network latency + cost per embedding. Incompatible with offline-capable v1 design.

---

## RQ-006: FastAPI Async Integration with Sync Libraries

**Question**: QdrantDB, sentence-transformers, and spaCy are all synchronous libraries. How do we integrate them correctly into a FastAPI async application?

**Decision**: Wrap all sync CPU/IO-bound calls with `asyncio.get_event_loop().run_in_executor(None, ...)` (default `ThreadPoolExecutor`). Use `aiosqlite` for all SQLite operations.

**Rules applied**:
| Library | Async strategy |
|---------|---------------|
| `aiosqlite` | Native async — use directly with `await` |
| `Qdrantdb` (sync client) | Wrap `.query()` and `.add()` in `run_in_executor` |
| `sentence_transformers` | Wrap `.encode()` in `run_in_executor` |
| `spaCy` / LangChain chunker | Wrap in `run_in_executor` |
| `pdfplumber` | Wrap in `run_in_executor` |

**Rationale**: FastAPI is built on Starlette's asyncio event loop. Blocking the event loop with sync I/O or CPU work causes all concurrent requests to stall. `run_in_executor(None, ...)` offloads to the default thread pool, freeing the event loop. This pattern satisfies the constitution's async-first requirement and the 10-concurrent-query success criterion (SC-006).

**Key implementation detail**: Instantiate `SentenceTransformer`, `Qdrantdb.Client`, and `spacy.load()` once at application startup (FastAPI `lifespan` context manager) and store as app-state singletons. Do not re-instantiate per request — model loading is expensive (1–3 seconds each).

---

## RQ-007: LLM Provider Abstraction

**Question**: How do we implement a provider-agnostic LLM interface that is swappable via configuration?

**Decision**: Define a `LLMProvider` abstract base class / Protocol with a single `async def chat(messages, tools)` method. Implement `OpenAIAdapter` as the default. `LLMFactory.from_config()` reads `LLM_PROVIDER` and `LLM_API_KEY` from environment and returns the correct adapter.

**Provider protocol**:
```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse: ...
```

**Environment variables**:
| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `openai` (default) \| `anthropic` \| `groq` |
| `LLM_MODEL` | e.g., `gpt-4o`, `claude-3-5-sonnet-20241022` |
| `LLM_API_KEY` | Provider API key |
| `LLM_BASE_URL` | Optional — for Azure OpenAI or local Ollama |

**Alternatives considered**:
- LangChain `BaseChatModel`: provides abstraction but couples the codebase to LangChain's versioning. Using the protocol directly allows LangChain to be used internally by one adapter while another uses raw `openai` SDK.
- Hardcoding OpenAI: violates constitution Principle I (provider-agnostic) and makes testing harder.

---

## Summary of Resolved Decisions

| ID | Decision | Status |
|----|----------|--------|
| RQ-001 | SQLite FTS5 for sparse BM25; QdrantDB for dense | ✅ Resolved |
| RQ-002 | `SpacyTextSplitter`, `chunk_size=400`, `chunk_overlap=75`, `en_core_web_sm` | ✅ Resolved |
| RQ-003 | LangChain `create_tool_calling_agent` + `AgentExecutor`; 5 defined tools | ✅ Resolved |
| RQ-004 | RRF with `k=60`; top-20 from each source, return top-10 fused | ✅ Resolved |
| RQ-005 | `all-MiniLM-L6-v2`, 384-dim, batch encode at ingestion, `run_in_executor` at query time | ✅ Resolved |
| RQ-006 | `run_in_executor` for sync libs; `aiosqlite` for SQLite; singleton model init at startup | ✅ Resolved |
| RQ-007 | `LLMProvider` Protocol; `LLMFactory.from_config()`; env-driven | ✅ Resolved |

No NEEDS CLARIFICATION items remain. Phase 1 design can proceed.
