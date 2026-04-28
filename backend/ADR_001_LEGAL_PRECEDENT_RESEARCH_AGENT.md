# Architecture Decision Record (ADR) 001
## Legal Precedent Research Agent — Design Rationale & Strategic Choices

**Scope**: Core agent architecture, retrieval strategy, query routing, scalability path  
**Related Documents**: [spec.md](specs/001-legal-precedent-research-agent/spec.md) | [plan.md](specs/001-legal-precedent-research-agent/plan.md) | [data-model.md](specs/001-legal-precedent-research-agent/data-model.md)

---

## Executive Summary

Legal precedent research assistant built on a **stateful graph-based reasoning workflow** that retrieves from an **embedded hybrid-search corpus** (dense semantic + sparse BM25) and synthesizes **IRAC-informed case analysis** with confidence-driven iterative refinement. The design prioritises:

- **Structured legal reasoning**: IRAC framework (Issue, Rules, Application, Conclusion) with precedent strength scoring and contradiction detection
- **Self-improving retrieval**: Reflection loop evaluates confidence (0–1) and refines retrieval strategy up to 2 iterations before synthesis
- **Accuracy over speed**: Hybrid retrieval (RRF-fused dense + sparse) + hierarchical chunking for precise context
- **Single-server simplicity**: SQLite + QdrantDB (embedded, zero infrastructure) for v1
- **Transparent reasoning trace**: Every planning, retrieval, reasoning, and reflection step is emitted as a timestamped event for auditability

This ADR documents **why** these choices were made, **what tradeoffs** were accepted, and **how the system would evolve** under different constraints.

---

## I. Architectural Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │   Ingest     │  │   Retrieval  │  │  Graph Workflow  │    │
│  │   Pipeline   │  │   (Hybrid)   │  │  (5-Node Engine) │    │
│  │              │  │              │  │                  │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
│       │                  │                    │               │
│       │                  │                    │               │
│       ▼                  ▼                    ▼               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │        Storage Layer (Repositories)                     │   │
│  │  • DocumentRepository                                   │   │
│  │  • ChunkRepository (hierarchical: parent + child)       │   │
│  │  • IngestionRunRepository                               │   │
│  │  • IngestionFailureRepository                           │   │
│  └────────────────────────────────────────────────────────┘   │
│                 │                      │                      │
│      ┌──────────▼──────────┐  ┌────────▼─────────┐            │
│      │   SQLite (aiosqlite)│  │ QdrantDB (Vector)│            │
│      │  • metadata         │  │ • embeddings     │            │
│      │  • chunk text       │  │ • dense search   │            │
│      │  • FTS5 index       │  │                  │            │
│      │  • BM25 search      │  │                  │            │
│      └─────────────────────┘  └──────────────────┘            │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │        Runtime Singletons (lazy-loaded at startup)   │    │
│  │  • SentenceTransformer (embeddings)                  │    │
│  │  • LLM Client (OpenAI / Groq / Azure)                │    │
│  │  • spaCy Model (sentence segmentation)               │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow: Three Bounded Contexts

#### 1. **Ingestion** (One-time or batch)
```
PDF corpus → Parser (pdfplumber) → Chunker (section-aware) → Embedder 
  → Storage (SQLite + QdrantDB + FTS5 index)
```

#### 2. **Retrieval** (Per query, deterministic)
```
Sub-query (from Planner) → DenseRetriever (QdrantDB) + SparseRetriever (SQLite FTS5) 
  → RRF Fusion (k=60) → Top-n Ranked Chunks
  → Parent Context Expansion
```

#### 3. **Reasoning Workflow** (Per query, stateful 5-node graph)
```
Query arrives
  ↓
[1] Conversational Fast Path? (heuristic patterns)
    ├─ YES → Direct LLM response (no retrieval)
    └─ NO  → Continue to Planner
  ↓
[2] PlannerNode    → query_type + depth + sub_queries + legal_issues
  ↓
    Is conversational strategy? → YES → skip to Direct Response
                               → NO  → Continue
  ↓
[3] RetrievalNode  → multi-query corpus search (deterministic, no LLM)
  ↓
[4] ReasonerNode   → IRAC reasoning + precedent strength scores (0–1) + contradiction detection
  ↓
[5] ReflectorNode  → confidence (0–1) + needs_more_retrieval?
    │
    ├─ confidence < 0.6 AND loop_count < 2 AND has refinement_queries?
    │  └─ YES → update plan.sub_queries, loop back to [3] (RetrievalNode)
    │
    └─ NO → Continue to Synthesis
  ↓
[6] SynthesisNode  → IRAC-informed PrecedentAnalysis JSON
                     (supporting/adverse split based on precedent_strengths)
                     + streamed narrative response
```

---

## II. Why This Architecture?

### A. Graph Workflow: Five Explicit Nodes (Not Tool-Calling Agent)

**Decision**: Replace dynamic tool-calling with a stateful, acyclic graph of five specialised nodes (Planner, Retrieval, Reasoner, Reflector, Synthesis).

**Why Nodes, Not Tool-Calling?**

The original tool-calling design gave the LLM autonomy to decide what to search. Early prototyping revealed problems:

1. **Unpredictable latency**: The LLM might make 3–6 tool calls per query; some queries converged faster, others meandered.
2. **Token waste**: Repeating context ("here's the query, here's what you retrieved so far, what's next?") across multiple LLM rounds consumed tokens inefficiently.
3. **Difficult debugging**: When the agent missed a case, it was unclear whether it was a search strategy failure (should have queried differently) or a reasoning failure (had the case but didn't recognise it).
4. **Auditability gap**: Lawyers couldn't easily audit *why* the system stopped searching—the LLM just decided.

**New Model**: Deterministic, auditable phases:

- **Planner** (1 LLM call): Decomposes the query once into sub_queries + legal_issues + depth; all downstream phases use these sub_queries (not the LLM's real-time decisions).
- **Retrieval** (0 LLM calls): Runs the sub_queries against the corpus deterministically; same query always produces same results.
- **Reasoner** (1 LLM call): Applies IRAC framework to the context; outputs structured reasoning + precedent strength scores.
- **Reflector** (1 LLM call): Self-assesses confidence + identifies gaps; **if** confidence < 0.6 and refinement_queries are available, suggests up to 2 more sub_queries.
- **Synthesis** (1–2 LLM calls): Produces final structured analysis or narrative, emitting streamed tokens.

**Tradeoff Accepted**: Fixed overhead of Planner (always one call, even if the query is trivial) vs. dynamic tool calling that might need 0 calls.
- **Mitigated by**: Conversational fast path (greetings skip the entire graph).
- **Latency trade**: Up to ~10s added overhead vs. potential 30s+ variation in tool-calling loop.
- **Benefit**: Predictable, auditable, testable.

**Alternatives Rejected**:
- **Tool-calling agent**: See "Problems" above.
- **LangGraph with explicit edges**: Functionally equivalent to nodes, but LangGraph adds 3–4 new concepts (interrupts, resume, branching); explicit nodes are simpler.
- **Hardcoded if-else pipeline**: No query decomposition (Planner) and no reflection loop (Reflector).

**Key Difference from Old ADR**:
The old ADR said "LLM dynamically selects search strategies." The new design says "Planner statically decomposes the query; Reflector dynamically refines if needed." The shift from *per-call autonomy* to *structural decomposition + targeted refinement* enables auditability without losing adaptivity.

---

### B. IRAC-Structured Reasoning with Precedent Strength Scoring

**Decision**: 
- **ReasonerNode** applies the IRAC framework (Issue, Rules, Application, Conclusion).
- For each precedent in the retrieved context, score its legal strength (0–1).
- Detect contradictions between precedents (e.g., two cases ruling differently on the same issue).
- Output is a structured `IRACReasoning` object (issue, applicable_rules, application, preliminary_conclusion, precedent_strengths, contradictions).

**Why IRAC?**

IRAC is the standard Indian legal reasoning framework taught in law schools:
- **Issue**: What is the precise legal question?
- **Rules**: What statutes, precedents, and principles apply?
- **Application**: How do those rules apply to *these* facts?
- **Conclusion**: What is the likely outcome?

Structuring the agent's reasoning in IRAC:
1. **Enforces rigour**: The LLM can't skip steps; it must articulate the issue and rules before application.
2. **Produces auditable output**: Each part is explicit JSON, not buried in prose.
3. **Guides subsequent synthesis**: Precedent strength scores (0–1) automatically split supporting (≥0.5) vs. adverse (<0.5) precedents.
4. **Enables contradiction detection**: If two precedents conflict on the same rule, the ReasonerNode flags it; SynthesisNode can highlight this for the lawyer.

**Precedent Strength Scoring**:

Each precedent is scored by the LLM using context:
- How factually similar is the precedent to the user's query?
- Does it support the user's legal position or undermine it?
- How binding is it (e.g., Supreme Court vs. district court)?

Score 0.8–1.0 = strong support
Score 0.5–0.8 = moderate or mixed
Score 0.0–0.5 = adverse or tangential

**Fallback (if LLM fails)**: Derive strength from retrieval relevance_score (0–1); assumes higher retrieval confidence = stronger precedent.

**Tradeoff Accepted**: One more LLM call (Reasoner) vs. deferring reasoning to SynthesisNode.
- **Benefit**: Separates retrieval quality assessment (Reflector uses precedent_strengths to decide if more retrieval is needed) from final synthesis.
- **Cost**: ~8 seconds for IRAC reasoning.

**Alternatives Rejected**:
- **No explicit IRAC step**: Reasoning is buried in the final synthesis LLM call; harder to debug and audit.
- **Scoring based solely on retrieval relevance**: Misses nuanced legal judgments (e.g., a Supreme Court ruling with lower retrieval score may be more legally binding).

**Contradiction Detection Example**:
```
Precedent A (Case X): "Unlicensed driver presumed negligent."
Precedent B (Case Y): "Lack of license doesn't establish negligence if factual circumstances show diligence."

ReasonerNode detects: These two precedents conflict on the same legal rule.
SynthesisNode uses this to tell the lawyer: "We have conflicting guidance. 
Case X argues strong presumption; Case Y argues defensibility despite license status. 
Your case facts are more aligned with Case Y's fact pattern."
```

---

### C. Reflection Loop: Iterative Refinement with Confidence Threshold

**Decision**: 
- **ReflectorNode** evaluates the IRAC reasoning and produces a `ReflectionResult` (confidence: 0–1, missing_aspects, needs_more_retrieval: bool, refinement_queries: list).
- If confidence < 0.6 AND loop_iteration < 2 AND refinement_queries are available, the workflow loops back to RetrievalNode with the new sub_queries.
- Otherwise, proceed to SynthesisNode.

**Why Reflection with Loop Budget?**

Legal research is inherently iterative:
1. Initial search might retrieve mainstream cases.
2. Lawyer reads IRAC analysis, sees gaps (e.g., "We have precedents on unlicensed drivers, but none on insurance liability in this specific scenario").
3. Lawyer refines search: "Show me cases combining driver negligence + insurer's duty."

Automating this loop:
- **ReflectorNode** asks the LLM: "Given this IRAC reasoning and these 10 precedents, what's missing?"
- LLM returns: "We have driver liability but not insurance liability. Recommend searching for 'insurance + negligent driver' or 'insurer's indemnity duty'."
- **RetrievalNode** runs the new searches.
- **ReasonerNode** re-scores precedents with the new context.
- **ReflectorNode** re-evaluates (second iteration).

**Confidence Threshold < 0.6**:
- 0.0–0.3 = Very uncertain; loop recommended.
- 0.3–0.6 = Moderate uncertainty; loop recommended if gap is addressable.
- 0.6–1.0 = Confident; proceed to synthesis.

**Loop Budget = 2 Max**:
- First iteration: initial retrieval + reasoning.
- Second iteration (if needed): refined retrieval + reasoning.
- After second iteration: always proceed to synthesis (avoid infinite refinement).

**Tradeoff Accepted**: Up to 3 more LLM calls (Reflector × 2 + Reasoner × 2) + latency cost of second retrieval pass.
- **Benefit**: Reduces "hallucinated" precedents (the agent genuinely retrieves and reasoned about them) and improves precision.
- **Mitigation**: Loop budget caps latency; _MAX_REFLECTION_LOOPS = 2 keeps typical query response ≤ 90s.

**Alternatives Rejected**:
- **No loop**: Single pass through retrieval + reasoning; faster but misses context gaps.
- **Unbounded loop**: LLM decides when to stop; risk of infinite refinement or analysis paralysis.
- **LLM autonomy (tool-calling)**: Loses determinism; hard to predict when refinement ends.

---

### D. Retrieval Strategy: Hybrid Dense + Sparse with RRF Fusion (Deterministic Multi-Query)

**Decision**: 
- **Dense retrieval** (QdrantDB + `all-MiniLM-L6-v2` embeddings, 384-dim)
- **Sparse retrieval** (SQLite FTS5 with BM25 ranking)
- **Merge** via Reciprocal Rank Fusion (k=60)
- **Multi-query**: Planner specifies sub_queries (e.g., ["unlicensed driver liability", "insurance denial", "motor accident negligence"]); RetrievalNode runs each against hybrid retriever, accumulates results, deduplicates by document_id, keeps top-15 unique documents.

**Why Hybrid?** (Same as old ADR — this part unchanged)

Legal documents have a dual nature:
1. **Semantic similarity** (dense): "insurance liability" ≈ judgments on insurer obligations.
2. **Keyword precision** (sparse): "Section 123 of IPC" or case names must match exactly.

**Why Deterministic Multi-Query (Not LLM-Driven Tool Calls)?**

- **Planner already decomposed**: If Planner output includes sub_queries, use them; don't ask LLM to decide per-call.
- **Reproducibility**: Same query → same sub_queries → same retrieval results. Easier to debug.
- **Efficiency**: All sub_queries run in parallel (asyncio.gather); slower than one sequential tool call, but faster than sequential LLM + tool + LLM + tool.
- **Rate-limit safety**: Groq API limits ~150 req/min per user; parallel sub_queries on corpus (no LLM) avoids eating rate-limit quota.

**Example Flow**:
```
Query: "Build a case for Mrs. Lakshmi Devi's unlicensed driver insurance claim."

Planner output:
  sub_queries: [
    "unlicensed driver insurance liability",
    "motor accident negligence insurer denial",
    "traffic violation personal injury compensation"
  ]

RetrievalNode:
  → search_corpus(sub_queries[0]) → 12 results
  → search_corpus(sub_queries[1]) → 12 results
  → search_corpus(sub_queries[2]) → 12 results
  → combine & deduplicate by document_id
  → keep top-15 unique documents by relevance_score

ReasonerNode:
  → IRAC analysis of these 15 documents
```

**Tradeoff Accepted**: ~200 ms per sub_query search × 3 sub_queries = ~600 ms retrieval vs. LLM calling tools dynamically (faster per-call, but may need 4–6 calls).
- **Net effect**: Similar latency, but predictable and parallelizable.

**RRF Formula** (unchanged from old ADR):
$$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{sparse}}(d)}$$

---

### E. Chunking Strategy: Hierarchical Structure-Aware with Section Detection

**Decision**: 
- **Structure detection**: Regex patterns identify legal sections (Facts, Issues, Arguments, Findings, Judgment).
- **Hierarchical**: Parent chunks (~2,000 chars) for LLM reasoning context; child chunks (~700 chars) for retrieval.
- **Sentence-boundary**: spaCy sentence segmentation ensures no mid-sentence breaks.
- **Overlap**: 20% overlap at both parent and child levels to preserve context continuity.

**Why Hierarchical?** (Same reasoning as old ADR — no change)

The core tension in legal RAG:
- Small chunks = precise retrieval, insufficient context.
- Large chunks = ample context, noisy retrieval.

**Solution**: Retrieve child, expand to parent for synthesis.

**Structure-Aware Difference from Old ADR**:

Old ADR mentioned "spaCy sentence segmentation"; new implementation adds regex patterns for section headers:
- Facts, Issues, Arguments, Findings, Judgment, etc.
- Each section is split into parents, each parent into children.
- Every chunk carries metadata: `section` (e.g., "facts", "findings") + `chunk_type` ("parent" or "child").

**Benefit**: LLM can reason about section-level context. A rule extracted from the Findings section is likely more binding than a supporting statement in Arguments.

**Fallback**: If no sections detected, treat entire document as one section ("other").

---

## III. Query Routing & Mode Classification

### Decision Tree (Conversational Fast Path → Planning → Workflow)

```
Query arrives
  │
  ├─→ [Heuristic Fast Path] Is conversational? (greeting, small-talk, etc.)
  │   ├─ YES ("hi", "hello", "thanks", "what can you do", …)
  │   │   └─ Direct LLM response (no retrieval, no graph)
  │   │      "I'm Casey, an Indian legal research assistant. I help analyze…"
  │   │
  │   └─ NO → Continue to Planner
  │
  └─→ [Planner] Decompose query
      ├─ Returns: query_type + strategy + sub_queries + legal_issues + depth
      │
      ├─ Strategy == "conversational"? (fallback, if Planner deems it conversational)
      │  └─ YES → Direct LLM response
      │
      ├─ Query type == "general_query"? (exploratory, not strategic)
      │  └─ YES → Retrieve (Retrieval Node)
      │           → Skip IRAC (Reasoner/Reflector/Synthesis)
      │           → Stream direct answer from LLM
      │
      └─ Query type == "precedent_research"? (strategic, needs case analysis)
         └─ YES → Full graph pipeline:
                  [2] Retrieval Node
                  [3] Reasoner Node (IRAC)
                  [4] Reflector Node (confidence & refinement loop)
                  [5] Synthesis Node (IRAC-informed JSON + narrative)
```

### Planner Output: Query Decomposition

**Example Input**: "Build a case for Mrs. Devi's unlicensed driver insurance claim. The insurer is denying liability."

**Planner Output**:
```json
{
  "query_type": "precedent_research",
  "strategy": "multi_step_research",
  "depth": "medium",
  "sub_queries": [
    "unlicensed driver insurance liability",
    "motor accident negligence insurer denial",
    "motor vehicle act insurance duty"
  ],
  "legal_issues": [
    "Does unlicensed driver status establish insurer liability?",
    "Can insurer deny indemnity based on license violation?"
  ]
}
```

**Key Differences from Old ADR**:
- Old ADR: "Tool-calling agent decides search dynamically."
- New: "Planner decides all sub_queries upfront."
- **Benefit**: Transparency (lawyer can see exactly what searches will run), parallelism (run all sub_queries at once), reproducibility (same query = same plan).

### Query-Type Classification

**Conversational** (detected by heuristic fast path or Planner fallback):
- Patterns: "hi", "hello", "thanks", "bye", "what can you do", "how are you"
- Response: Direct LLM reply, no corpus involvement.

**General Query** (Planner decides OR fallback keywords):
- Keywords: "which", "list", "summarise", "explain", "describe" (exploratory).
- *No* keywords: "strategy", "precedent", "adverse", "support our case", "argue", "win", "likelihood", "risk".
- Response: Retrieve from corpus, stream direct answer grounded in retrieved documents.
- **Example**: "Which judgments involve commercial vehicles?" → retrieve cases with vehicles, answer directly.

**Precedent Research** (Planner decides OR fallback keywords):
- Keywords: "strategy", "precedent", "adverse precedent", "support my case", "support our case", "argue that", "case law", "claim compensation", "litigation strategy", "will I win", "how strong is our case", "help me argue", "find cases for", "find precedents", "distinguish", "distinguish this case".
- Response: Full IRAC-informed pipeline; output is structured `PrecedentAnalysis` (supporting_precedents, adverse_precedents, strategy_recommendation) + streamed narrative.
- **Example**: "Build a case for this motor accident with an unlicensed driver." → retrieve, reason, reflect, synthesize into supporting/adverse analysis + strategy.

### Planner Fallback (If LLM Unavailable)

```python
def _fallback_plan(query_text: str) -> PlannerOutput:
    lowered = query_text.lower()
    
    # Follow-up references to prior retrieval = always general
    is_follow_up = any(phrase in lowered for phrase in [
        "which of these", "these judgments", "those cases",
        "the above cases", "from the results", …
    ])
    
    # Only strategic keywords signal precedent_research
    is_research = not is_follow_up and any(phrase in lowered for phrase in [
        "strategy", "precedent", "adverse precedent",
        "support my case", "support our case", "argue that", …
    ])
    
    return PlannerOutput(
        query_type="precedent_research" if is_research else "general_query",
        requires_retrieval=True,
        depth="medium" if is_research else "shallow",
        sub_queries=[query_text],  # fallback: single query
        legal_issues=[query_text],
        strategy="multi_step_research" if is_research else "direct_answer",
    )
```

**Key Insight**: The fallback is keyword-based and deterministic. If the LLM Planner fails or is unavailable, the system degrades gracefully by using a simple heuristic.

---

## IV. Strategic Tradeoffs & Justification

### Tradeoff 1: Stateful Graph Workflow vs Tool-Calling Agent

| Aspect | Graph Workflow (Chosen) | Tool-Calling (Rejected) |
|---|---|---|
| **Latency predictability** | ~45–90s (fixed phases) | 30–150s (variable LLM calls) |
| **Sub-query determination** | Upfront (Planner once) | Dynamic (LLM per round) |
| **Parallelism** | All sub_queries run together | Sequential tool calls |
| **Auditability** | Every phase emits events; full trace visible | Tool calls logged but LLM reasoning opaque |
| **Confidence loop** | Explicit Reflector with budget (max 2 loops) | Unbounded agent autonomy; unclear when to stop |
| **Token efficiency** | Planner (1 call) sets context for all downstream | Repeating context across multiple LLM rounds |
| **Testability** | Each node has deterministic input/output | Tool-calling nondeterministic (LLM variance) |

**Justified for v1** because:
- Lawyers need predictable response times and transparent reasoning.
- Parallelisable retrieval (all sub_queries at once) is faster than sequential LLM + tool + LLM.
- Event emission enables real-time progress UI ("Reasoning… [progress indicator]").
- Reflection loop with budget prevents "analysis paralysis."

**Migration path**: If early users want deeper autonomy (e.g., "let the agent decide how many searches to run"), replace Reflector with a conditional that checks LLM-generated confidence and dynamically extends budget. The graph structure remains unchanged; only ReflectorNode logic is modified.

---

### Tradeoff 2: SQLite + QdrantDB (Embedded) vs Managed Vector DB

| Aspect | Embedded (Chosen) | Managed (Rejected) |
|---|---|---|
| **Vector DB** | QdrantDB embedded | MongoDB Atlas Vector Search / Pinecone |
| **Sparse search** | SQLite FTS5 | Elasticsearch or PostgreSQL |
| **Setup time** | ~2 minutes (no account needed) | ~30 minutes (cloud signup, credentials, VPC) |
| **Infrastructure cost** | $0 (runs on server) | ~$57–300/month (Atlas tier depends on scale) |
| **BM25 support** | ✓ SQLite FTS5 native | Varies (Elasticsearch has it; Atlas doesn't) |
| **ACID guarantees** | ✓ SQLite transactions | Eventual consistency (Atlas) |
| **Offline capability** | ✓ Entire system offline | ✗ Depends on cloud API |
| **Data residency** | On-premise | Cloud provider (US region by default) |
| **Max corpus** | ~50k docs (~1.5GB index) | Unlimited |
| **Concurrent queries** | ~10 before degradation | 100+ (with auto-scaling) |

**Justified for v1** because:
- Legal teams are small (1–5 lawyers); 10 concurrent queries is ample.
- Corpus is stable (uploaded once, indexed once); not real-time ingestion.
- Privacy/data residency is a hard constraint in India (lawyers' case data on-premise preferred).
- Simplicity reduces time-to-market and operational risk.

**Scaling path at 5,000 documents**:
- QdrantDB query latency rises ~2x (load on single node).
- FTS5 queries still acceptable but I/O contention with concurrent ingestions.
- **Solution**: Migrate vector DB to MongoDB Atlas Vector Search, FTS to PostgreSQL with pg_trgm (trigram index). Repositories layer abstracts both; minimal code changes.

**Note on QdrantDB vs ChromaDB**:
- Old plan mentioned ChromaDB; implementation uses QdrantDB.
- Both are embedded vector DBs; QdrantDB was chosen for better performance and memory efficiency.
- Swap is transparent to application (VectorStore interface).

---

### Tradeoff 3: Hierarchical Chunking (Parent + Child) vs Single-Layer Chunks

| Aspect | Hierarchical (Chosen) | Flat (Rejected) |
|---|---|---|
| **Storage overhead** | 1.5x (parents + children both stored) | 1.0x |
| **Retrieval precision** | High (small child chunks) | Medium (fixed-size chunks misalign with sections) |
| **Synthesis context quality** | High (parent provides full section) | Low (isolated snippet, mid-argument breaks) |
| **Chunking complexity** | Medium (parent → child mapping) | Low (fixed-size loop) |
| **Expansion latency** | ~10ms (join query) | N/A |
| **LLM reasoning quality** | Higher (section-level context) | Lower (fragmented facts) |

**Justified** because:
- Legal reasoning requires surrounding context (the full Findings section, not just one sentence).
- Small retrieval units (700 chars) prevent noisy context; parents (2000 chars) provide full reasoning.
- Storage cost (1.5x) is negligible for 50k docs (~1.5 GB additional, acceptable on modern servers).

**Hierarchy Example**:
```
Document: "Case ABC v. XYZ" (Motor accident judgment)

Parent chunk (section="findings"):
  "The findings of this court are as follows:
   1. The driver held a valid international driving permit but did not carry it.
   2. The insurer raised objection to indemnity based on license violation.
   3. However, Indian courts have held that mere absence of a license at the moment
      of driving is not absolute bar to recovery if the driver had valid credentials.
   4. Applying this principle to the facts, we hold the insurer liable."

Child chunk 1 (section="findings", parent_id=parent_chunk.id):
  "The driver held a valid international driving permit but did not carry it.
   The insurer raised objection to indemnity based on license violation."

Child chunk 2 (section="findings", parent_id=parent_chunk.id):
  "However, Indian courts have held that mere absence of a license
   is not absolute bar to recovery if the driver had valid credentials."

When retrieving:
  → Search corpus for "unlicensed driver insurance"
  → Retrieve [child_chunk_2] (high relevance)
  → Expand to [parent] for synthesis
  → LLM gets full section context, not isolated snippet
```

---

### Tradeoff 4: Reflection Loop (Max 2 Iterations) vs Single Pass

| Aspect | Reflection Loop (Chosen) | Single Pass (Rejected) |
|---|---|---|
| **Iterations** | Up to 2 (initial + 1 refinement) | 1 |
| **Typical latency** | 45–90s (usually 1 iteration) | 30–45s |
| **Precedent coverage** | Higher (gaps addressed) | Lower (may miss nuanced cases) |
| **Confidence score use** | Active (confidence < 0.6 → loop) | Passive (confidence reported only) |
| **User experience** | "Thinking…" (longer, but more thorough) | "Instant" (faster, less thorough) |
| **LLM cost** | Higher (~5 calls for research query) | Lower (~3 calls) |
| **Failure gracefully** | Loop budget prevents infinite refinement | Always fast response |

**Justified** because:
- Legal research is inherently iterative; initial results often reveal gaps.
- Precedent research (vs. general queries) can afford +30–45s latency for better coverage.
- Loop budget (max 2) prevents analysis paralysis.
- Confidence threshold (< 0.6) is data-driven; only loops if genuinely uncertain.

**When Loop Activates**:
```
ReflectorNode evaluates IRAC reasoning:
  "confidence = 0.55 (moderate uncertainty)
   missing_aspects = ['insurance liability', 'compensation estimates']
   refinement_queries = ['insurance indemnity duty', 'compensation precedents']
   needs_more_retrieval = true"

RetrievalNode runs refinement_queries:
  → 2nd iteration retrieval & reasoning
  
ReflectorNode re-evaluates (2nd iteration):
  "confidence = 0.78 (confident)
   needs_more_retrieval = false"

→ Proceed to Synthesis
```

**When Loop Doesn't Activate**:
- Confidence ≥ 0.6 (high confidence in initial reasoning).
- loop_count ≥ 2 (budget exhausted; stop regardless).
- refinement_queries is empty (LLM couldn't identify gaps).

---
| Cost | Included in server | $57/month minimum (Atlas cluster) |

**Justified for v1** because:
- Spec explicitly prioritises simplicity (Constitution Principle V).
- FTS5 is battle-tested, built into Python's sqlite3 (zero dependency).
- QdrantDB handles dense search adequately for this corpus size.
- Single-server v1 doesn't need cloud resilience features.

**Migration path**: Repositories layer (`storage/repositories.py`) abstracts storage; swapping backends requires only changing one module.

---

### Tradeoff 3: Real-Time Index Streaming vs Batch Ingestion

| Approach | Real-Time (Rejected) | Batch (Chosen) |
|---|---|---|
| Ingestion trigger | Upload PDF → index immediately | Operator initiates /ingest endpoint |
| Latency | Per-document: 5–10 seconds | Bulk: parallel chunking + embedding |
| Error handling | Partial corpus state (complex rollback) | All-or-nothing per run_id (simple) |
| User experience | Instant feedback per document | Operator polls /ingest/{run_id} status |
## V. How the System Scales: 50 Documents → 5,000 Documents

### Bottleneck Analysis at 5,000 Documents

| Metric | 50 Docs | 5,000 Docs | Bottleneck |
|---|---|---|---|
| Chunks | ~2,500 | ~250,000 | Vector index size |
| Vector index size (384-dim) | ~15 MB | ~1.5 GB | Single-node storage |
| Embedding time (ingestion) | 10 seconds | 20–30 minutes | CPU (even with batching) |
| Dense search latency | ~150 ms | ~300 ms (+ index load) | QdrantDB query time grows with index size |
| SQLite FTS5 query latency | ~50 ms | ~200 ms | BM25 ranking over 250k docs |
| Synthesis LLM latency | ~8 seconds | ~10 seconds | Token count (larger context window) |
| **Typical query response** | **≤ 90s** | **≤ 180s** | Agent loops + synthesis latency |

### Architectural Changes Needed at 5,000 Documents

#### 1. **Distributed Vector Search**
**Problem**: QdrantDB embedding index (~1.5 GB) + query latency → becomes bottleneck.

**Solution**: Migrate to managed vector database.
- **Option A**: MongoDB Atlas Vector Search ($57–300/month, depending on tier)
- **Option B**: Weaviate / Pinecone (cloud) or Milvus (self-hosted)

**Implementation**:
```python
# Current (SQLite FTS5 + QdrantDB)
retriever = Retriever(session_factory, vector_store, embedder)

# Future (MongoDB Atlas Vector Search)
retriever = Retriever(
    session_factory,
    vector_store=MongoDBAtlasVectorStore(...),
    embedder=embedder
)
# Repositories layer unchanged
```

#### 2. **Distributed Sparse Search**
**Problem**: SQLite on a single server handles ~250k documents, but concurrent queries compete for I/O.

**Solution**: Move FTS5 index to a read-optimized store.
- **Option A**: PostgreSQL + pg_trgm (trigram index) for substring search
- **Option B**: Elasticsearch (BM25 at scale)

**Implementation**: Minimal—same interface, different backend.

```python
sparse_retriever = SparseRetriever(
    session_factory=session_factory,
    # Swapped backend: PostgreSQL instead of SQLite
)
```

#### 3. **Parallel Ingestion**
**Problem**: Embedding 250k chunks sequentially takes 20–30 minutes; CPU thread pool is starved.

**Solution**: 
- Pre-allocate embedding workers (e.g., `max_workers=8` ThreadPoolExecutor).
- Batch encode chunks in groups of 128.
- Stream embeddings to QdrantDB in batches.

```python
async def batch_embed(chunks, batch_size=128):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = await asyncio.get_event_loop().run_in_executor(
            executor, embedder.encode, batch
        )
        yield embeddings
```

#### 4. **Query Result Caching**
**Problem**: Repeated queries on the same legal issues should not require re-running the entire agent loop.

**Solution**: 
- Cache top-20 results per query (keyed by query embedding).
- TTL: 1 day (corpus may be re-ingested).

```python
cache = RedisCache(ttl=86400)  # 1 day
key = hash_query(query_embedding)
if cache.get(key):
    return cache.get(key)
```

#### 5. **Agent Iteration Limits**
**Problem**: At 5,000 docs, retrieval is slower; agent may take 6 iterations to converge.

**Solution**: 
- Tighter early-exit thresholds: stop at ≥15 chunks instead of ≥20.
- Reduced `max_iterations` from 6 to 4.

```python
_MIN_CHUNKS_FOR_SYNTHESIS = 15  # was 20
_MAX_ITERATIONS = 4              # was 6
```

---

## VI. What Would Change With Another Week?

### Priority 1: Cross-Encoder Re-Ranking (1–2 days)

**Current bottleneck**: RRF fusion is effective but sometimes retrieves tangential cases.

**Solution**: Add a cross-encoder stage after hybrid fusion.

```
Dense + Sparse → RRF top-20 → Cross-Encoder (e.g., ms-marco-MiniLM-L-6) 
  → Re-rank by relevance score → Return top-10
```

**Latency cost**: +200 ms per query (cross-encoder inference).  
**Precision gain**: ~5–10% improvement in Precision@10.  
**Implementation**: 3-line integration into retriever.

---

### Priority 2: Persistent Chat History & Multi-Turn Reasoning (2–3 days)

**Current state**: Conversation history is stored but not actively used by the agent.

**Enhancement**: 
- Use full conversation history to contextualize each new query.
- Let the agent reference prior retrieved cases in follow-up queries.

**Example**:
```
User Q1: "Tell me about precedents for unlicensed driver liability."
         → Agent retrieves 5 cases

User Q2: "Given those cases, what's our best argument?"
         → Agent refers back to Q1's retrieved cases + new synthesis
```

**Implementation**:
- Pass conversation history to agent as system context.
- Modify synthesis to accept prior retrieved docs as seeds.

---

### Priority 3: Domain-Specific Embedding Fine-Tuning (2–3 days)

**Current**: `all-MiniLM-L6-v2` is general-purpose; trained on 1B sentence pairs but not legal-domain-specific.

**Enhancement**: Fine-tune embeddings on Indian legal judgments.

**Dataset needed**: 
- Positive pairs: (query, relevant judgment chunks) — collect from user interactions.
- Negative pairs: (query, irrelevant chunks) — mine from corpus.

**Tool**: Hugging Face `sentence-transformers` fine-tuning pipeline.

**Expected improvement**: +15–25% Precision@10 on legal queries.

---

### Priority 4: Structured Reasoning Trace Visualization (1–2 days)

**Current**: Agent steps are logged but not visualized.

**Enhancement**: Build a frontend panel showing:
- Agent's decision tree (which searches ran, in what order).
- Confidence scores at each phase.
- Why precedents were classified as supporting/adverse.

**Value**: Lawyers can audit reasoning; easier debugging of missed cases.

---

### Priority 5: Adverse Precedent Distinguishing (1–2 days)

**Current**: Adverse precedents are flagged, but distinguishing arguments are generic.

**Enhancement**: 
- Add a specialized LLM prompt that takes an adverse precedent + client facts.
- Generates fact-specific distinguishing arguments.

```
System Prompt:
  "Given an adverse precedent and the client's factual situation, 
   propose 2–3 legally sound distinguishing arguments."

Example:
  Adverse: "Driver without valid license = presumed negligent"
  Client: "Driver had valid international license (IDP)"
  → Distinguishing: "Court in precedent dealt with zero license; 
                     here driver had international credential, 
                     arguing reciprocal validity..."
```

---

## VII. Known Limitations & Acceptance Criteria

### Limitation 1: Hallucinated Citations
**Risk**: Agent may cite a judgment ID that does not exist in the corpus.

**Mitigation**: 
- Every citation in output is validated against the database before returning.
- Failed citations are stripped with a note: "Could not verify in corpus."

---

### Limitation 2: Context Window Saturation
**Risk**: At 5,000 docs, synthesis LLM context may exceed token limits.

**Acceptance Criterion**: Limit context to top-15 deduped chunks; if >15, the 16th is discarded.
```python
_MAX_CONTEXT_CHUNKS = 15
```

---

### Limitation 3: Semantic Drift on Out-of-Domain Queries
**Risk**: If a lawyer asks a non-legal question (e.g., "What's the weather?"), the agent may return incorrect legal information.

**Mitigation**: 
- Conversational detector catches off-topic greetings.
- Agent system prompt warns: "Only answer legal questions about the indexed corpus."
- If no relevant documents are found, agent returns "No relevant precedents found."

---

### Limitation 4: Cold Start with Empty Corpus
**Risk**: User runs queries before ingestion completes.

**Acceptance Criterion**: API returns explicit error: "Corpus not yet indexed. Please trigger /api/v1/ingest first."

---

## VIII. Conclusion

### Design Philosophy

AI prioritises **accuracy and transparency** over raw speed:

- **Accuracy**: Hybrid retrieval (RRF) beats either dense-only or sparse-only.
- **Transparency**: Every tool call and reasoning step is logged and auditable.
- **Simplicity**: Single-server embedded stack (SQLite + QdrantDB) eliminates DevOps burden for v1.
- **Autonomy**: LLM-driven agent workflow handles diverse queries without hardcoded branching.

### Migration Path to Scale

| Phase | Corpus Size | Architecture |
|---|---|---|
| **v1 (Current)** | 50–100 docs | SQLite + QdrantDB (embedded) |
| **v2** | 500–5k docs | PostgreSQL FTS + MongoDB Atlas Vector Search |
| **v3** | 5k–100k docs | Elasticsearch + Pinecone / Weaviate |
| **v4** | 100k+ docs | Multi-region distributed (SaaS) |

### Critical Dependencies

1. **OpenAI-compatible LLM provider** (OpenAI, Azure OpenAI, Groq, Llama 2 via API).
2. **Corpus stability**: Assume ingestion is infrequent (batch-based).
3. **Legal text structure**: Judgments must follow Indian court judgment conventions (Facts → Issues → Findings → Judgment); highly unstructured PDFs require re-chunking.

### Recommended Next Steps

1. **Monitor agent loop convergence** on live queries; tune `_MAX_ITERATIONS` and `_MIN_CHUNKS_FOR_SYNTHESIS` based on empirical latency.
2. **Collect user feedback** on precedent classifications (supporting vs adverse); use to fine-tune query classifier.
3. **Plan cross-encoder integration** for v1.5 (medium-effort, high-value precision gain).
4. **Begin embedding fine-tuning dataset collection** now (can be done in parallel with v1 deployment).

