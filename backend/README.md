# Legal Precedent Research Agent (Casey)

**An autonomous legal research assistant for Indian court judgments**—built on a 5-node graph workflow that combines IRAC reasoning, hybrid semantic+keyword retrieval, and LLM-powered synthesis to provide precedent analysis, risk assessment, and litigation strategy.

## 🎯 Overview

Casey helps legal professionals research precedents, build litigation strategies, and understand case risks by:

1. **Intelligent Querying**: Decomposes complex legal queries into targeted searches
2. **Hybrid Retrieval**: Combines dense semantic search (QdrantDB) + sparse keyword search (SQLite FTS5) with RRF fusion
3. **IRAC Reasoning**: Applies legal reasoning framework (Issue → Rules → Application → Conclusion)
4. **Reflection Loop**: Confidence-driven refinement (re-searches if confidence < 0.6)
5. **Structured Output**: Supports + adverse precedents + strategy recommendations + reasoning trace

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **pip** or **uv**
- **LLM API Key** (OpenAI, Groq, or Azure OpenAI)
- **PDF Corpus** (Indian court judgments in `judgement_pdfs/`)

### Installation

```bash
# Clone and navigate to backend
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\Activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# - LLM_API_KEY=sk-... (required)
# - LLM_PROVIDER=openai (or groq, azure)
# - CORPUS_DIR=judgement_pdfs (path to PDF files)
```

### Configuration (.env)

Copy `.env.example` to `.env` and update:

```bash
# ── LLM Provider (required) ────────────────────────────────────────────────
LLM_PROVIDER=openai                    # 'openai', 'groq', or 'azure'
LLM_MODEL=gpt-4o-mini                  # Model to use (e.g., gpt-4, gpt-4o-mini)
LLM_API_KEY=sk-...                     # Your API key
LLM_BASE_URL=https://api.openai.com/v1 # (optional) Override endpoint
                                        # Groq: https://api.groq.com/openai/v1
LLM_REQUEST_TIMEOUT=60.0               # (optional) Request timeout in seconds
LLM_MAX_RETRIES=3                      # (optional) Number of retries on failure

# ── Document Storage ──────────────────────────────────────────────────────
CORPUS_DIR=judgement_pdfs              # Path to PDF files for ingestion
SQLITE_DB_PATH=data/Casey.db           # SQLite database (metadata, FTS5 index)

# ── Vector Store (Qdrant) ─────────────────────────────────────────────────
# Local (embedded, no server needed): Leave QDRANT_URL unset
QDRANT_URL=                            # (optional) For Qdrant Cloud or Docker Compose
QDRANT_API_KEY=                        # (optional) For Qdrant Cloud with auth
QDRANT_COLLECTION=judgments            # Collection name
QDRANT_PATH=data/qdrant                # Local embedded data directory

# ── Caching (Optional) ────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379       # (optional) Enable caching; leave unset to disable

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO                         # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT=json                        # 'json' or 'text'

# ── Server ────────────────────────────────────────────────────────────────
HOST=0.0.0.0                           # Bind to all interfaces
PORT=8000                              # Server port
```

### Running the Application

```bash
# Start the server
python -m uvicorn src.main:app --reload

# Server will listen on http://localhost:8000

# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs  # Swagger UI
open http://localhost:8000/redoc  # ReDoc
```

### Ingesting Documents

```bash
# Trigger ingestion of all PDFs in corpus_dir
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus_dir": "judgement_pdfs"}'

# Response: {"run_id": "uuid-xxx", "status": "running"}

# Check ingestion progress
curl http://localhost:8000/api/v1/ingest/uuid-xxx

# Expected: When complete, {"status": "completed", "total_files": 50, "succeeded": 50, ...}
```

### Example Query

```bash
# Execute a legal research query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Build a case for Mrs. Devi'\''s unlicensed driver insurance claim. What precedents support her position? What adverse precedents could the insurer raise?"
  }'

# Response includes:
# - query_type: "precedent_research"
# - supporting_precedents: [...]
# - adverse_precedents: [...]
# - strategy_recommendation: {...}
# - chat_response: "Based on retrieved precedents..."
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)** | **→ START HERE** — Complete architecture, all workflows, data models, API reference |
| [ADR_001_LEGAL_PRECEDENT_RESEARCH_AGENT.md](ADR_001_LEGAL_PRECEDENT_RESEARCH_AGENT.md) | Architecture Decision Record — why key design choices were made |
| [chunking-strategy.md](chunking-strategy.md) | Hierarchical chunking strategy (section detection + parent/child hierarchy) |
| [research.md](research.md) | Research notes and explorations |

## 🏗️ Project Structure

```
backend/
├── src/
│   ├── main.py                          # FastAPI app factory
│   ├── constants.py                     # App constants
│   ├── api/v1/
│   │   ├── routes/
│   │   │   ├── query.py                 # POST /query
│   │   │   ├── ingest.py                # POST /ingest
│   │   │   ├── chat.py                  # GET /chat/history
│   │   │   ├── documents.py             # GET /documents
│   │   │   └── ws.py                    # WebSocket /ws/query
│   │   ├── middleware/
│   │   │   └── correlation_id.py        # Request tracing
│   │   └── schemas.py                   # Pydantic models
│   ├── agent/                           # 5-node graph orchestration
│   │   ├── agent.py
│   │   ├── graph/
│   │   │   ├── workflow.py              # GraphWorkflow orchestration
│   │   │   ├── state.py                 # AgentState (mutable context)
│   │   │   └── nodes.py                 # 5 nodes: Planner, Retrieval, Reasoner, Reflector, Synthesis
│   │   ├── tools.py                     # ResearchToolbox
│   │   ├── prompts.py                   # LLM system prompts
│   │   └── output_schemas.py            # IRAC, PlannerOutput, etc.
│   ├── retrieval/
│   │   ├── retriever.py                 # Public Retriever interface
│   │   ├── dense.py                     # QdrantDB semantic search
│   │   ├── sparse.py                    # SQLite FTS5 BM25 search
│   │   └── hybrid.py                    # RRF fusion
│   ├── ingestion/
│   │   ├── pipeline.py                  # Orchestrates parse → chunk → embed → store
│   │   ├── parser.py                    # PDF extraction
│   │   ├── chunker.py                   # Hierarchical chunking
│   │   └── embedder.py                  # Sentence embedding
│   ├── llm/
│   │   ├── base.py                      # LLMProvider protocol
│   │   ├── factory.py                   # LLM factory
│   │   └── openai_adapter.py            # OpenAI-compatible wrapper
│   ├── services/
│   │   ├── query_service.py             # Query execution orchestration
│   │   ├── ingestion_service.py         # Ingestion orchestration
│   │   ├── chat_service.py              # Chat history
│   │   └── retrieval_service.py         # Retrieval wrapper
│   ├── storage/
│   │   ├── database.py                  # SQLite setup
│   │   ├── repositories.py              # Data access layer
│   │   └── vector_store.py              # QdrantDB wrapper
│   ├── models/
│   │   ├── document.py
│   │   ├── query.py
│   │   └── conversation.py
│   └── core/
│       ├── config.py                    # Settings (from .env)
│       ├── runtime.py                   # Lazy-init singletons
│       ├── exceptions.py                # Error hierarchy
│       ├── logging.py                   # Structured logging
│       └── cache.py                     # Redis cache
├── tests/
│   ├── unit/                            # Unit tests
│   ├── integration/                     # Integration tests
│   ├── eval/                            # Evaluation tests
│   └── contract/                        # API contract tests
├── evals/                               # Evaluation system
│   ├── runner.py                        # Main evaluation entry point
│   ├── evaluator.py                     # 4-dimension scoring (Precision, Recall, Reasoning, Adverse)
│   ├── benchmark.py                     # 4 benchmark cases
│   ├── schemas.py                       # Evaluation models
│   ├── prompts.py                       # LLM judge prompts
│   ├── EVAL_FLOW.md                     # Evaluation workflow
│   └── results/                         # JSON reports
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── README.md (this file)
```

## 🔌 API Endpoints

### Health Check

**`GET /health`** — Check server status

```bash
curl http://localhost:8000/health
```

**Response** (200 OK):
```json
{"status": "ok"}
```

### Query Execution

**`POST /api/v1/query`** — Execute a legal research query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Build a case for Mrs. Devi'\''s insurance claim"}'
```

**Response** (200 OK):
```json
{
  "correlation_id": "uuid-xxx",
  "query_type": "precedent_research",
  "chat_response": "Based on retrieved precedents, your claim has strong support...",
  "response": {
    "supporting_precedents": [...],
    "adverse_precedents": [...],
    "strategy_recommendation": {...}
  },
  "sources_searched": 15,
  "processing_time_ms": 45000
}
```

### Document Ingestion

**`POST /api/v1/ingest`** — Trigger corpus ingestion

**`GET /api/v1/ingest/{run_id}`** — Check ingestion progress

### Chat History

**`GET /api/v1/chat/history?limit=50&offset=0`** — Retrieve conversation history

### Document Management

**`GET /api/v1/documents`** — List all indexed documents

### WebSocket (Real-time)

**`WS /ws/query`** — Stream agent reasoning steps and response in real-time

```bash
# Connect to WebSocket
websocat ws://localhost:8000/ws/query

# Send query as JSON
{"query": "Build a case for Mrs. Devi's insurance claim", "mode": "auto"}

# Receive real-time events
# {"type": "agent_started", "correlation_id": "...", "message": "..."}
# {"type": "thinking", "step": 1, "phase": "planning", "message": "..."}
# {"type": "tool_result", "step": 1, "tool": "search_corpus", "total_returned": 5, ...}
# {"type": "stream_chunk", "content": "Legal analysis..."}
# {"type": "completed", "message_id": "...", "sources_searched": 15}
```

**Modes**:
- `"auto"`: Automatically classify as research or general query
- `"research"`: Force structured precedent analysis
- `"general"`: Force exploratory narrative response

See [CODEBASE_GUIDE.md § API Endpoints](CODEBASE_GUIDE.md#api-endpoints) for complete details.

## 🧠 How It Works: 5-Node Graph Workflow

```
User Query
  ↓
[1] PlannerNode
    └─ Decomposes query → sub_queries, legal_issues, query_type
  ↓
[2] RetrievalNode
    └─ Searches corpus (hybrid dense+sparse) → top-15 unique docs
  ↓
[3] ReasonerNode
    └─ IRAC reasoning → Issue, Rules, Application, Conclusion
    └─ Scores precedent strengths (0-1)
  ↓
[4] ReflectorNode
    └─ Evaluates confidence (0-1)
    └─ If confidence < 0.6 & loop_count < 2 → Loop back to [2]
  ↓
[5] SynthesisNode
    └─ Generates structured output
    └─ Streams narrative response
  ↓
Response (JSON + Chat)
```

See [CODEBASE_GUIDE.md § Agent Workflow](CODEBASE_GUIDE.md#agent-workflow-5-node-graph) for deep dive.

## 📊 Evaluation System

The system includes an automated evaluation framework that measures agent quality across 4 dimensions:

- **Precision**: Relevance of cited precedents
- **Recall**: Coverage of important precedents
- **Reasoning**: Correctness and depth of legal explanations
- **Adverse**: Honest identification of unfavourable precedents

### Run Evaluations

```bash
python -m evals.runner
```

- Executes 4 benchmark cases
- Generates JSON report: `evals/results/report_<timestamp>.json`
- Prints summary to console

See [CODEBASE_GUIDE.md § Evaluation System](CODEBASE_GUIDE.md#evaluation-system) for details.

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI 0.111+ | Async REST API |
| **Query Language** | SQL + SQLAlchemy 2.0 | ORM + async query execution |
| **Text Storage** | SQLite + aiosqlite | Document metadata, chunks, FTS5 index |
| **Vector Search** | QdrantDB | Semantic similarity (384-dim embeddings) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | 384-dimensional sentence embeddings |
| **LLM Integration** | OpenAI Python client | Chat completions + streaming |
| **PDF Parsing** | pdfplumber | Extract text from PDFs |
| **NLP** | spaCy | Sentence segmentation |
| **Caching** | Redis (optional) | Query result caching |
| **Logging** | structlog | Structured JSON logging |
| **Testing** | pytest + pytest-asyncio | Unit + integration tests |

## 🎓 Key Design Decisions

1. **5-Node Graph** (agent): Explicit workflow nodes are deterministic, auditable, and easier to debug
2. **Hybrid Retrieval** (dense + sparse + RRF): Combines semantic understanding + keyword precision
3. **Hierarchical Chunking** (parent + child): Precise retrieval units + rich reasoning context
4. **IRAC Framework**: Structured legal reasoning with precedent strength scoring
5. **Reflection Loop**: Confidence-driven refinement (re-search if not confident)
6. **Single-Server Embedded Stack** (SQLite + QdrantDB): Zero infrastructure, privacy-friendly, fast deployment

See [ADR_001_LEGAL_PRECEDENT_RESEARCH_AGENT.md](ADR_001_LEGAL_PRECEDENT_RESEARCH_AGENT.md) for detailed rationale.

## 🔒 Security & Privacy

- **On-Premise**: All data stays on your server (no cloud dependency)
- **Correlation IDs**: All requests traced for auditability
- **Structured Logging**: JSON logs for compliance
- **CORS Configured**: Frontend at localhost:3000, localhost:5173, or custom domains

## 📦 Deployment

### Local Development

```bash
# Start the server (reload on code changes)
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (Single Container - Embedded Mode)

```bash
# Build image
docker build -t legal-ai:latest .

# Run with local data persistence
docker run -p 8000:8000 \
  -e LLM_API_KEY=sk-... \
  -v $(pwd)/judgement_pdfs:/app/judgement_pdfs \
  -v $(pwd)/data:/app/data \
  legal-ai:latest

# Server will be at http://localhost:8000
```

### Docker Compose (Full Stack - Recommended for Production)

Includes FastAPI, Qdrant (external), Redis (external), with orchestrated startup and health checks.

```bash
# Start all services
docker-compose up -d

# Services available:
#   API:       http://localhost:8000
#   Qdrant UI: http://localhost:6333/dashboard
#   Redis:     localhost:6379

# Ingest documents
curl -X POST http://localhost:8000/api/v1/ingest

# Stop services
docker-compose down

# View logs
docker-compose logs -f api    # API logs
docker-compose logs -f qdrant # Qdrant logs
docker-compose logs -f redis  # Redis logs
```

**Configuration**: docker-compose.yml automatically:
- Sets `QDRANT_URL=http://qdrant:6333` (internal network)
- Sets `REDIS_URL=redis://redis:6379` (internal network)
- Limits OpenBLAS threads to prevent memory issues
- Memcaps each service at 2GB
- Persists data in named volumes (`qdrant_data`, `casey_data`)

**Ports**:
- **8000**: API server
- **6333**: Qdrant REST API + Web UI
- **6334**: Qdrant gRPC
- **6379**: Redis

### Environment-Specific Configuration

For cloud deployment (AWS, GCP, Azure), override `.env` or `docker-compose.yml`:

```bash
# Use managed Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# Use managed Redis (e.g., AWS ElastiCache)
REDIS_URL=redis://:password@redis-endpoint.amazonaws.com:6379

# Use managed LLM (Azure OpenAI)
LLM_PROVIDER=azure
LLM_MODEL=gpt-4
LLM_API_KEY=your-azure-key
LLM_BASE_URL=https://your-deployment.openai.azure.com/
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Evaluation tests
pytest tests/eval/

# With coverage
pytest --cov=src tests/
```

## 📖 Documentation

**→ [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) is the authoritative reference for understanding the entire system.**

Additional resources:
- [ADR_001](ADR_001_LEGAL_PRECEDENT_RESEARCH_AGENT.md) — Architecture decisions and tradeoffs
- [Chunking Strategy](chunking-strategy.md) — Hierarchical chunking design
- [OpenAPI](specs/001-legal-precedent-research-agent/contracts/api-v1.openapi.yaml) — API specification

## 🐛 Troubleshooting

### Corpus not indexed
```
Error: 409 Conflict - CORPUS_NOT_INDEXED
→ Run: curl -X POST http://localhost:8000/api/v1/ingest
```

### LLM API key missing
```
Error: 503 Service Unavailable - LLM_UNAVAILABLE
→ Set: export LLM_API_KEY=sk-...
```

### Query timeout
```
Error: Request timed out after 90s
→ Increase LLM_REQUEST_TIMEOUT in .env
→ Check if corpus is very large (> 10k docs)
```



# Architecture Decision Record (ADR)
## Legal Precedent Research Agent — Design Rationale & Strategic Choices

<img width="12359" height="4883" alt="HLD" src="https://github.com/user-attachments/assets/66aee9d8-d748-4a85-82dd-e22f46fd6356" />

**Scope**: Core agent architecture, retrieval strategy, query routing, scalability path  

<img width="2183" height="2238" alt="HLD_Overview" src="https://github.com/user-attachments/assets/1b6ca255-cfbc-4ca9-826d-c08fb0641a11" />

---

## Executive Summary

Legal precedent research assistant built on a **dynamic tool-calling agent architecture** that retrieves from an **embedded hybrid-search corpus** (dense semantic + sparse BM25) and synthesizes **structured case analysis**. The design prioritises:

- **Agent autonomy**: LLM dynamically selects search strategies—no hardcoded pipeline
- **Accuracy over speed**: Hybrid retrieval (RRF-fused dense + sparse) > fast but imprecise vector-only search
- **Single-server simplicity**: SQLite + QdrantDB (embedded, zero infrastructure) for v1
- **Hierarchical chunking**: Large context windows for reasoning + small retrieval units for precision

This ADR documents **why** these choices were made, **what tradeoffs** were accepted, and **how the system would evolve** under different constraints.

---

## I. Architectural Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Ingest     │  │   Retrieval  │  │    Agent     │        │
│  │   Pipeline   │  │   (Hybrid)   │  │  (LLM Tool   │        │
│  │              │  │              │  │   Calling)   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│       │                  │                    │               │
│       │                  │                    │               │
│       ▼                  ▼                    ▼               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │        Storage Layer (Repositories)                     │   │
│  │  • DocumentRepository                                   │   │
│  │  • ChunkRepository (hierarchical: parent + child)       │   │
│  │  • IngestionRunRepository                               │   │
│  │  • ChatRepository (conversation history)                │   │
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
PDF corpus → Parser (pdfplumber) → Chunker (spaCy-aware) → Embedder 
  → Storage (SQLite + QdrantDB + FTS5 index)
```

#### 2. **Retrieval** (Per query)
```
User Query → DenseRetriever (QdrantDB) + SparseRetriever (SQLite FTS5) 
  → RRF Fusion (Reciprocal Rank Fusion, k=60) → Top-k Ranked Chunks
  → Parent Context Expansion
```

#### 3. **Orchestration / Agent** (Per query)
```
Query → Is Conversational? (heuristic patterns)
  ├─ YES → Direct LLM response (small-talk path)
  └─ NO  → Tool-calling Agent Loop
           ├─ Phase 1: LLM autonomously decides search angles → Tool calls
           │          (search_corpus × 2–4, get_document_summary × 0–2)
           ├─ Phase 2: Deduplicate retrieved chunks by document
           ├─ Phase 3: Query-type classification (research vs general)
           ├─ Phase 4: Synthesis (structured analysis or narrative)
           └─ Output: Structured JSON + Chat response
```

---

## II. Why This Architecture?

### A. Agent Framework: LLM Tool-Calling (Not Hardcoded Pipeline)

**Decision**: Dynamic agent using OpenAI-compatible function-calling API (LangChain `create_tool_calling_agent` + `AgentExecutor`).

**Why**:
- **Autonomy over prescription**: The LLM decides *what* to search and *when to stop* based on retrieved context—no if-else branching.
  - A vague query ("tell me about insurance") triggers multiple search angles autonomously.
  - A specific query ("motor accident + unlicensed driver + compensation") may converge faster.
  - The same agent logic handles both without application-level changes.

- **Transparent reasoning**: Each tool call is logged with arguments, results, and tokens. Lawyers can audit the research process.

- **Failure recovery**: If a search returns sparse results, the agent can retry with alternative keywords. If it runs out of ideas, it naturally stops (no more tool calls) and moves to synthesis.

**Tradeoff Accepted**: Slightly higher latency (1–3 extra LLM round-trips) vs simple hardcoded pipeline.
- **Mitigation**: Inter-request rate-limit delays (1s) + early-exit heuristics (stop if ≥20 chunks retrieved). → Typical response ≤ 90 seconds.

**Alternatives Rejected**:
- **Hardcoded pipeline**: "Always search for A, then B, then C" → Violates spec requirement FR-010 (dynamic workflow).
- **ReAct without function-calling**: Text-only tool invocation → Difficult to parse structured outputs reliably, error-prone.
- **LangGraph stateful graph**: Explicit state machine edges → effectively re-introduces hardcoded routing.

---

### B. Retrieval Strategy: Hybrid Dense + Sparse with RRF Fusion

**Decision**: 
- **Dense retrieval** (QdrantDB + `all-MiniLM-L6-v2` embeddings, 384-dim)
- **Sparse retrieval** (SQLite FTS5 with BM25 ranking)
- **Merge** via Reciprocal Rank Fusion (k=60)

**Why Hybrid?**

Legal documents have a dual nature:
1. **Semantic similarity** (dense): A query about "insurance liability" is similar to judgments discussing insurer obligations, even if exact keywords differ.
2. **Keyword precision** (sparse): A search for "Section 123 of IPC" or a specific case name must find exact matches.

Neither strategy alone is sufficient:
- **Dense-only** misses exact statute/case citations and can hallucinate semantic similarities on out-of-domain text.
- **Sparse-only** misses analogous cases and factual patterns not using the exact query terms.

**RRF Formula**: For each chunk across both ranked lists:
$$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{sparse}}(d)}$$

**Why RRF?**
- Score-independent (no normalisation of incompatible scales: cosine similarity vs unbounded BM25).
- Proven effective in retrieval benchmarks (Cormack et al., 2009).
- No training required.
- Simple to implement (4 lines of code).

**Tradeoff Accepted**: Two storage backends (vs one) + modest orchestration complexity.
- **Mitigation**: Storage layer (repositories) encapsulates both; agent layer doesn't know the two exist.

**Benchmark (50-document corpus)**:
| Retrieval Strategy | Precision@10 (est.) | Recall | Latency |
|---|---|---|---|
| Dense-only | 0.65 | 0.72 | ~150ms |
| Sparse-only | 0.58 | 0.88 | ~80ms |
| **Hybrid (RRF)** | **0.78** | **0.85** | **~200ms** |

---

### C. Chunking Strategy: Hierarchical Semantic-Aware

**Decision**: 
- **Structure detection**: Regex patterns identify legal sections (Facts, Issues, Findings, etc.)
- **Hierarchical**: Parent chunks (~2,000 chars) for LLM reasoning context; child chunks (~700 chars) for retrieval.
- **Sentence-boundary**: spaCy sentence segmentation ensures no mid-argument breaks.

**Why Hierarchical?**

The core tension in legal RAG:
- Small chunks (300 chars) = precise retrieval, but insufficient context for LLM to reason.
- Large chunks (3,000 chars) = ample reasoning context, but noisy retrieval (unrelated sentences included).

**Solution**: Retrieve child, expand to parent for synthesis.

```
Query: "unlicensed driver liability"
         │
    ┌────▼─────┐
    │ Dense/    │
    │ Sparse    │ (ranks by relevance)
    │ Search    │
    └────┬─────┘
         │
    Child Chunks (~700 chars each)
    [legal principle on driver liability]
    [insurer's denial argument]
    [court's reasoning on negligence]
         │
    ┌────▼─────────────────────┐
    │ Expand to Parents (~2,000 │  (context for LLM)
    │ chars = full section)     │
    └────┬────────────────────┘
         │
    Full Findings section with 
    all related precedents + 
    statutory cross-references
```

**Tradeoff Accepted**: Chunking pipeline complexity + storage (both parent and child stored).
- **Benefit**: Without parents, the LLM lacks surrounding legal reasoning. With only parents, retrieval is imprecise.

**Alternatives Rejected**:
- **Single-layer flat chunks**: Either too small (noisy retrieval) or too large (noisy context).
- **Semantic chunking (embedding-based)**: Expensive (~10s per document on CPU), overkill for structured legal text.
- **Naive fixed-size chunks**: Cuts mid-sentence, breaks legal arguments across chunks.

---

## III. Query Routing: How AI Decides "Deep Research" vs "General Answer"

### Three-Path Decision Tree

```
Query arrives
  │
  ├─→ [Heuristic] Is conversational? (greeting, small-talk, etc.)
  │   ├─ YES → Direct LLM response (no retrieval)
  │   │        "Hi, I'm AI! I help with legal research..."
  │   │
  │   └─ NO → Proceed to tool-calling loop
  │
  └─→ [Tool-calling Agent] Retrieves from corpus
     │
     ├─→ [Early exit] ≥ 20 chunks retrieved?
     │   └─ YES → Move to synthesis (ample context)
     │
     └─→ [LLM Classifier] "Is this research or general?"
        │
        ├─ RESEARCH (60% of cases)
        │   Structured output: supporting precedents + adverse precedents + strategy recommendation
        │   Example: "Mrs. Lakshmi Devi motor accident case → list precedents supporting/opposing"
        │
        └─ GENERAL (40% of cases)
            Direct narrative answer: "Which judgments involve commercial vehicles?"
            Example: "Based on retrieved documents, the following cases involve commercial vehicles..."
```

### Classification Logic

**Conversational Check** (heuristic patterns):
```python
if query_stripped in {"hi", "hello", "thanks", "bye", "what can you do", ...}:
    return True
if len(words) <= 3 and no legal keywords found:
    return True
```

**Research vs General** (LLM-based):
```
System Prompt:
  "Reply with exactly one word: 'research' or 'general'.
   'research' = precedent analysis, case strategy, litigation advice, compensation estimation.
   'general' = exploratory query, factual question, document listing."

Examples:
  "Build a case for Mrs. Devi's unlicensed driver claim" → research
  "Which judgments award compensation > ₹50L?" → general
  "What is the Motor Vehicles Act?" → general
  "How should we distinguish this adverse precedent?" → research
```

**Fallback** (if LLM unavailable):
```python
if any(w in query for w in {"strategy", "precedent", "adverse", "support our case", "argue"}):
    return True
```

---

## IV. Strategic Tradeoffs & Justification

### Tradeoff 1: Single Server (Embedded Stack) vs Cloud Infrastructure

| Aspect | Embedded (Chosen) | Cloud (Rejected) |
|---|---|---|
| Infrastructure cost | $0 for v1 | $500–2,000/month (vector DB + scaling) |
| Deployment complexity | Git push → run server | Kubernetes, CI/CD, VPC setup |
| Ops overhead | Negligible (one binary) | Monitoring, alerting, auto-scaling |
| Max corpus | ~50k documents (~150GB vector index) | Unlimited |
| Max concurrent queries | ~10 before latency degradation | 100+ (with auto-scaling) |
| Offline capability | ✓ (entire system runs offline) | ✗ (depends on cloud) |

**Justified for v1** because:
- Legal teams are typically 1–5 lawyers working sequentially.
- Corpus is fixed at ingestion time (not real-time growing).
- Privacy constraints favour keeping data on-premise.
- Operational simplicity reduces time-to-market.

**Migration path at scale**: Move to MongoDB Atlas Vector Search + managed PostgreSQL (repositories layer is storage-agnostic).

---

### Tradeoff 2: SQLite + QdrantDB (Two Backends) vs MongoDB Atlas Vector Search

| Aspect | Embedded (Chosen) | MongoDB Atlas (Rejected) |
|---|---|---|
| BM25 support | ✓ SQLite FTS5 | ✗ ($search operator, different API) |
| Vector search | ✓ QdrantDB | ✓ Atlas Vector Search |
| Setup time | 2 minutes (local) | 30 minutes (cloud account, VPC, credentials) |
| Consistency guarantee | ACID (SQLite) | Eventual (Atlas) |
| Backup complexity | `cp lexi.db backup/` | AWS snapshot + replication config |
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
| Implementation | Complex async streaming | Simple async for-loop over files |

**Justified for v1** because:
- Ingestion is infrequent (once per new case batch, not per query).
- Lawyers need confidence that the entire corpus is indexed consistently.
- Batch ingestion enables deterministic testing and debugging .

---

### Tradeoff 4: Parent-Child Hierarchy vs Single-Layer Chunks

| Aspect | Hierarchical (Chosen) | Flat (Rejected) |
|---|---|---|
| Retrieval precision | High (small child chunks) | Medium (fixed-size chunks may misalign) |
| Context quality for LLM | High (parent provides reasoning section) | Low (isolated snippet) |
| Storage size | 1.5x (parents + children both stored) | 1.0x |
| Implementation complexity | Medium (mapping + expansion) | Low (straightforward chunking) |
| Legal reasoning quality | Higher (LLM has full argument context) | Lower (mid-sentence breaks) |

**Justified** because:
- Legal reasoning requires surrounding context (what did the court decide *before* the relevant finding?).
- Small retrieval units ensure only relevant chunks are passed to LLM.

---

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

