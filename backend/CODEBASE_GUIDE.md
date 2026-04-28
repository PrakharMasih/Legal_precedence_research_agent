# Legal AI Backend - Comprehensive Codebase Guide

**Last Updated**: April 28, 2026  
**Project**: Legal Precedent Research Agent (Casey)  
**Purpose**: Understand the complete architecture, data flows, workflows, and how every component works together

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Core Components & Modules](#core-components--modules)
3. [Data Models & Database Schema](#data-models--database-schema)
4. [API Endpoints](#api-endpoints)
5. [Request Workflows](#request-workflows)
6. [Agent Workflow (5-Node Graph)](#agent-workflow-5-node-graph)
7. [Retrieval System](#retrieval-system)
8. [Ingestion Pipeline](#ingestion-pipeline)
9. [LLM Integration](#llm-integration)
10. [Error Handling](#error-handling)
11. [Configuration & Runtime](#configuration--runtime)
12. [Deployment & Lifecycle](#deployment--lifecycle)

---

## System Architecture Overview

### High-Level Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                            │
│                  (Async Request Handler)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              REST API Endpoints (v1)                     │    │
│  │  POST /api/v1/query         POST /api/v1/ingest         │    │
│  │  GET /api/v1/chat/history   GET /api/v1/documents       │    │
│  │  WebSocket /ws/chat                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Service Layer (Business Logic)               │    │
│  │  ├─ QueryService          (execute queries)            │    │
│  │  ├─ IngestionService      (ingest PDFs)               │    │
│  │  ├─ ChatService           (manage history)             │    │
│  │  └─ RetrievalService      (search corpus)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │        Agent & Reasoning Layer                          │    │
│  │  ├─ LegalResearchAgent     (5-node graph orchestration)│    │
│  │  ├─ PlannerNode            (query decomposition)       │    │
│  │  ├─ RetrievalNode          (multi-query search)        │    │
│  │  ├─ ReasonerNode           (IRAC reasoning)            │    │
│  │  ├─ ReflectorNode          (confidence evaluation)      │    │
│  │  └─ SynthesisNode          (final response generation) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │        Data Access Layer (Repositories)                 │    │
│  │  ├─ DocumentRepository     (document metadata)         │    │
│  │  ├─ ChunkRepository        (document chunks)           │    │
│  │  ├─ ChatRepository         (conversation history)       │    │
│  │  └─ IngestionRunRepository (ingestion tracking)        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Storage Backends                             │    │
│  │  ├─ SQLite (aiosqlite)                                 │    │
│  │  │  ├─ documents table       (metadata)                │    │
│  │  │  ├─ chunks table          (text + hierarchy)        │    │
│  │  │  ├─ messages table        (chat history)            │    │
│  │  │  ├─ ingestion_runs table  (run tracking)            │    │
│  │  │  └─ FTS5 index            (BM25 sparse search)      │    │
│  │  │                                                      │    │
│  │  ├─ QdrantDB (Vector Store)                            │    │
│  │  │  └─ Embeddings + Dense Vector Search                │    │
│  │  │                                                      │    │
│  │  └─ Redis (Optional Cache)                             │    │
│  │     └─ Query results, chat history, counts             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow (Simplified)

```
User Query
   ↓
[HTTP POST /api/v1/query]
   ↓
QueryService.execute_query()
   ↓
LegalResearchAgent.run()
   ├─ Conversational fast path? → Direct LLM response
   └─ No → GraphWorkflow.run()
      ├─ [1] PlannerNode           → decompose query
      ├─ [2] RetrievalNode         → search corpus
      ├─ [3] ReasonerNode          → IRAC reasoning
      ├─ [4] ReflectorNode         → confidence check + loop?
      └─ [5] SynthesisNode         → final response
   ↓
[Store messages in ChatRepository]
   ↓
[Return response to user]
   ↓
[HTTP 200 OK with QueryResponse]
```

---

## Core Components & Modules

### Directory Structure

```
src/
├── main.py                          # FastAPI app factory
├── constants.py                     # App-wide constants
│
├── api/
│   └── v1/
│       ├── routes/
│       │   ├── query.py             # POST /query endpoint
│       │   ├── ingest.py            # POST /ingest endpoint
│       │   ├── chat.py              # GET /chat/history endpoint
│       │   ├── documents.py         # GET /documents endpoint
│       │   └── ws.py                # WebSocket endpoint
│       ├── middleware/
│       │   └── correlation_id.py    # Correlation ID tracking
│       └── schemas.py               # Pydantic request/response models
│
├── agent/
│   ├── agent.py                     # LegalResearchAgent (public entry point)
│   ├── tools.py                     # ResearchToolbox (search_corpus, get_document_summary)
│   ├── prompts.py                   # LLM system prompts (Planner, IRAC, Reflection, etc.)
│   ├── output_schemas.py            # Pydantic models for IRAC, Planner, Reflection outputs
│   └── graph/
│       ├── workflow.py              # GraphWorkflow (orchestrates 5 nodes)
│       ├── state.py                 # AgentState (mutable context threaded through nodes)
│       └── nodes.py                 # PlannerNode, RetrievalNode, ReasonerNode, ReflectorNode, SynthesisNode
│
├── retrieval/
│   ├── retriever.py                 # Public Retriever interface
│   ├── dense.py                     # DenseRetriever (QdrantDB wrapper)
│   ├── sparse.py                    # SparseRetriever (SQLite FTS5 wrapper)
│   ├── hybrid.py                    # RRF fusion (rank fusion)
│
├── ingestion/
│   ├── pipeline.py                  # IngestionPipeline (orchestrates parsing → chunking → embedding → storage)
│   ├── parser.py                    # PDF parsing (pdfplumber)
│   ├── chunker.py                   # Hierarchical chunking (section-aware, parent/child)
│   ├── embedder.py                  # Embedding generation (SentenceTransformer)
│
├── llm/
│   ├── base.py                      # LLMProvider protocol
│   ├── factory.py                   # LLMFactory.from_config()
│   ├── openai_adapter.py            # OpenAI-compatible adapter (works with OpenAI, Groq, Azure)
│
├── services/
│   ├── query_service.py             # Orchestrates agent execution & message persistence
│   ├── ingestion_service.py         # Orchestrates ingestion pipeline
│   ├── chat_service.py              # Chat history management
│   └── retrieval_service.py         # Wrapper around Retriever
│
├── storage/
│   ├── database.py                  # SQLite connection pool, schema init
│   ├── repositories.py              # DocumentRepository, ChunkRepository, ChatRepository, etc.
│   └── vector_store.py              # QdrantDB wrapper
│
├── models/
│   ├── document.py                  # Document, Chunk Pydantic models
│   ├── query.py                     # RankedChunk, SearchMode models
│   ├── conversation.py              # Message model
│
├── core/
│   ├── config.py                    # Settings (reads from .env)
│   ├── runtime.py                   # RuntimeServices (lazy-init of all singletons)
│   ├── exceptions.py                # CaseyError, CorpusNotIndexedError, etc.
│   ├── logging.py                   # Structured JSON logging with correlation IDs
│   └── cache.py                     # Redis cache client
│
└── utils/
    ├── responses.py                 # Error response building
    ├── timestamps.py                # ISO timestamp utilities
```

### Key Design Patterns

| Pattern | Usage | Location |
|---------|-------|----------|
| **Repository Pattern** | Abstract data access | `storage/repositories.py` |
| **Service Layer** | Encapsulate business logic | `services/` |
| **Factory Pattern** | Create LLM provider | `llm/factory.py` |
| **Middleware** | Cross-cutting concerns (correlation IDs) | `api/v1/middleware/` |
| **Lazy Initialization** | Runtime singleton pattern | `core/runtime.py` |
| **Async-first** | All I/O is async (SQLite via aiosqlite, QdrantDB) | Throughout |
| **Node-based Workflow** | 5-node graph with mutable state | `agent/graph/` |

---

## Data Models & Database Schema

### Database Tables (SQLite)

#### 1. **documents** table
```sql
CREATE TABLE documents (
  id TEXT PRIMARY KEY,                   -- SHA256 hash of file
  file_name TEXT NOT NULL UNIQUE,
  case_name TEXT,
  court_name TEXT,
  judgment_date TEXT,
  page_count INT,
  char_count INT,
  ingested_at TEXT NOT NULL,             -- ISO format
  status TEXT DEFAULT 'success'          -- 'success', 'failed'
);
```

**Purpose**: Metadata for each ingested PDF document.

#### 2. **chunks** table
```sql
CREATE TABLE chunks (
  id TEXT PRIMARY KEY,                   -- UUID
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  char_start INT NOT NULL,
  char_end INT NOT NULL,
  chunk_index INT NOT NULL,
  embedded_at TEXT NOT NULL,
  section TEXT DEFAULT 'other',          -- 'facts', 'issues', 'findings', 'judgment', etc.
  chunk_type TEXT DEFAULT 'child',       -- 'parent', 'child'
  parent_id TEXT,                        -- References parent chunk (if this is a child)
  
  UNIQUE(document_id, chunk_index)
);
```

**Purpose**: Document content split into hierarchical chunks (parent + child).  
**Key insight**: Only child chunks are embedded in QdrantDB; parent chunks provide context for LLM reasoning.

#### 3. **messages** table
```sql
CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL,                    -- 'user', 'assistant'
  content TEXT NOT NULL,
  query_type TEXT,                       -- 'precedent_research', 'general_query', 'conversational'
  sources_searched INT,
  raw_response JSON,
  agent_steps JSON,
  created_at TEXT NOT NULL
);
```

**Purpose**: Persist conversation history for multi-turn context.

#### 4. **ingestion_runs** table
```sql
CREATE TABLE ingestion_runs (
  id TEXT PRIMARY KEY,
  corpus_dir TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  total_files INT DEFAULT 0,
  succeeded INT DEFAULT 0,
  failed INT DEFAULT 0,
  total_chunks INT DEFAULT 0,
  status TEXT DEFAULT 'running'          -- 'running', 'completed', 'failed'
);
```

**Purpose**: Track batch ingestion jobs.

#### 5. **ingestion_failures** table
```sql
CREATE TABLE ingestion_failures (
  id INT PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
  file_name TEXT NOT NULL,
  error_message TEXT
);
```

**Purpose**: Log errors during ingestion (e.g., corrupted PDFs).

#### 6. **FTS5 Virtual Table** (sparse BM25 index)
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  content,
  document_id UNINDEXED
);
```

**Purpose**: Enable full-text search with BM25 ranking for keyword-based retrieval.

### Vector Store (QdrantDB)

**Collection**: `judgments` (configurable via `QDRANT_COLLECTION`)

**Point Structure**:
```json
{
  "id": "chunk_id (UUID)",
  "vector": [0.123, 0.456, ..., 0.789],  // 384-dim (all-MiniLM-L6-v2)
  "payload": {
    "document_id": "doc_id",
    "chunk_index": 0,
    "section": "findings",
    "chunk_type": "child"
  }
}
```

**Purpose**: Enable semantic similarity search (dense retrieval).

### Pydantic Models

#### Document & Chunk Models
```python
class Document(BaseModel):
    id: str                              # SHA256 hash
    file_name: str
    case_name: str | None
    court_name: str | None
    judgment_date: str | None
    page_count: int
    char_count: int
    ingested_at: datetime
    status: Literal["success", "failed"]

class Chunk(BaseModel):
    id: str                              # UUID
    document_id: str
    content: str
    char_start: int
    char_end: int
    chunk_index: int
    embedded_at: datetime
    section: str                         # 'facts', 'issues', 'findings', etc.
    chunk_type: str                      # 'parent', 'child'
    parent_id: str | None
```

#### Query & Retrieval Models
```python
class RankedChunk(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    case_name: str | None
    content: str
    parent_content: str | None           # Expanded parent for LLM context
    section: str
    relevance_score: float               # 0.0–1.0
    rrf_score: float                     # RRF fusion score

class SearchMode(str, Enum):
    DENSE = "dense"                      # QdrantDB semantic search
    SPARSE = "sparse"                    # SQLite FTS5 keyword search
    HYBRID = "hybrid"                    # RRF fusion of dense + sparse
```

#### Agent Reasoning Models
```python
class PlannerOutput(BaseModel):
    query_type: str                      # 'precedent_research', 'general_query', 'conversational'
    requires_retrieval: bool
    depth: str                           # 'shallow', 'medium', 'deep'
    sub_queries: list[str]               # Diverse search angles
    legal_issues: list[str]              # Precise legal questions
    strategy: str                        # 'multi_step_research', 'direct_answer', 'conversational'

class IRACReasoning(BaseModel):
    issue: str
    applicable_rules: list[str]
    application: str
    preliminary_conclusion: str
    precedent_strengths: dict[str, float]  # document_id → 0.0–1.0
    contradictions: list[ContradictionNote]

class ReflectionResult(BaseModel):
    confidence: float                    # 0.0–1.0
    reasoning_quality: str               # 'sufficient', 'needs_improvement', 'insufficient'
    missing_aspects: list[str]
    needs_more_retrieval: bool
    refinement_queries: list[str]
    contradictions_addressed: bool

class PrecedentAnalysis(BaseModel):
    supporting_precedents: list[SupportingPrecedent]
    adverse_precedents: list[AdversePrecedent]
    strategy_recommendation: StrategyRecommendation
```

---

## API Endpoints

### 1. Query Endpoint

**Endpoint**: `POST /api/v1/query`

**Request**:
```json
{
  "query": "Build a case for Mrs. Devi's unlicensed driver insurance claim"
}
```

**Response (200 OK)**:
```json
{
  "correlation_id": "uuid-xxx",
  "query_type": "precedent_research",
  "chat_response": "Based on retrieved precedents, your claim has strong support...",
  "response": {
    "supporting_precedents": [
      {
        "document_id": "hash_xxx",
        "file_name": "judgment_001.pdf",
        "case_name": "Case A v. Case B",
        "excerpt": "...",
        "legal_principle": "Insurer liability established despite driver negligence...",
        "factual_alignment": "..."
      }
    ],
    "adverse_precedents": [
      {
        "document_id": "hash_yyy",
        "file_name": "judgment_002.pdf",
        "case_name": "Case C v. Case D",
        "excerpt": "...",
        "risk_description": "Court ruled insurer can deny liability for unlicensed drivers",
        "distinguishing_argument": "..."
      }
    ],
    "strategy_recommendation": {
      "priority_arguments": [
        "Distinguish Case C based on statutory context",
        "Emphasize Case A's precedent on insurer duty"
      ],
      "compensation_range": "₹50L–₹100L",
      "risks": [
        "Adverse precedent on license requirements",
        "Factual similarity to Case C raises counter-argument risk"
      ]
    }
  },
  "sources_searched": 15,
  "processing_time_ms": 45000,
  "user_message_id": "msg_xxx",
  "assistant_message_id": "msg_yyy"
}
```

**Error Responses**:
- `503 Service Unavailable`: LLM provider unreachable
- `409 Conflict`: Corpus not indexed
- `400 Bad Request`: Validation error

---

### 2. Ingestion Endpoints

#### Trigger Ingestion

**Endpoint**: `POST /api/v1/ingest`

**Request**:
```json
{
  "corpus_dir": "/path/to/pdfs"  // Optional; uses config default if omitted
}
```

**Response (202 Accepted)**:
```json
{
  "correlation_id": "uuid-xxx",
  "run_id": "run_xxx",
  "status": "running",
  "message": "Ingestion started for /path/to/pdfs"
}
```

#### Get Ingestion Status

**Endpoint**: `GET /api/v1/ingest/{run_id}`

**Response (200 OK)**:
```json
{
  "correlation_id": "uuid-xxx",
  "run_id": "run_xxx",
  "status": "completed",
  "corpus_dir": "/path/to/pdfs",
  "total_files": 50,
  "succeeded": 48,
  "failed": 2,
  "total_chunks": 2400,
  "failures": [
    {
      "file_name": "corrupted.pdf",
      "error_message": "PDF is image-only; no extractable text"
    },
    {
      "file_name": "malformed.pdf",
      "error_message": "Unexpected end of file"
    }
  ]
}
```

---

### 3. Chat History Endpoint

**Endpoint**: `GET /api/v1/chat/history?limit=50&offset=0`

**Response (200 OK)**:
```json
{
  "total": 120,
  "limit": 50,
  "offset": 0,
  "messages": [
    {
      "id": "msg_xxx",
      "role": "user",
      "content": "What is the Motor Vehicles Act?",
      "query_type": "general_query",
      "sources_searched": 5,
      "created_at": "2026-04-28T10:00:00Z"
    },
    {
      "id": "msg_yyy",
      "role": "assistant",
      "content": "The Motor Vehicles Act of 1988 is a key statute governing...",
      "query_type": "general_query",
      "sources_searched": 5,
      "created_at": "2026-04-28T10:00:30Z"
    }
  ]
}
```

---

### 4. Documents Endpoint

**Endpoint**: `GET /api/v1/documents`

**Response (200 OK)**:
```json
{
  "total": 50,
  "documents": [
    {
      "id": "sha256_hash",
      "file_name": "judgment_001.pdf",
      "case_name": "Mrs. Lakshmi Devi v. Insurance Co.",
      "court_name": "District Court, Chennai",
      "judgment_date": "2023-05-15",
      "page_count": 25,
      "char_count": 15000,
      "ingested_at": "2026-04-20T09:00:00Z",
      "status": "success"
    }
  ]
}
```

---

### 5. WebSocket Endpoint

**Endpoint**: `WebSocket /ws/chat`

**Purpose**: Real-time streaming of agent reasoning steps and final response.

**Message Flow**:
```
Client connects
   ↓
Client sends: {"type": "query", "text": "..."}
   ↓
Server streams multiple messages:
   ├─ {"type": "thinking", "phase": "planning", "message": "..."}
   ├─ {"type": "thinking", "phase": "retrieval", "message": "..."}
   ├─ {"type": "reasoning", "message": "Applying IRAC..."}
   ├─ {"type": "stream_chunk", "content": "Based on..."}
   ├─ {"type": "stream_chunk", "content": " retrieved "}
   ├─ {"type": "stream_chunk", "content": " precedents..."}
   └─ {"type": "done", "message": "Query complete"}
   ↓
Client receives real-time progress
```

---

## Request Workflows

### 1. Query Execution Workflow

```
[1] HTTP POST /api/v1/query {query: "..."}
         ↓
[2] Middleware: CorrelationIDMiddleware adds correlation_id header
         ↓
[3] route.submit_query() extracts correlation_id + payload
         ↓
[4] QueryService.execute_query()
         ├─ Build agent
         ├─ Get recent chat history
         ├─ Call agent.run()
         ├─ Store user + assistant messages in ChatRepository
         └─ Return result
         ↓
[5] Serialize to QueryResponse Pydantic model
         ↓
[6] HTTP 200 OK + JSON response
```

### 2. Ingestion Workflow

```
[1] HTTP POST /api/v1/ingest {corpus_dir: "..."}
         ↓
[2] IngestionService.start_ingestion()
         ├─ Check if ingestion already running
         ├─ Create IngestionRun record
         └─ Return run_id
         ↓
[3] Schedule async task: IngestionService.execute_ingestion(corpus_dir, run_id)
         ├─ For each PDF in corpus_dir:
         │  ├─ [a] parse_pdf()
         │  │    └─ Extract: text, case_name, page_count, file_hash
         │  ├─ [b] chunker.chunk_text()
         │  │    └─ Detect sections (Facts, Issues, Findings, etc.)
         │  │    └─ Create parent chunks (~2000 chars, 20% overlap)
         │  │    └─ Create child chunks (~700 chars, 20% overlap)
         │  ├─ [c] embedder.embed_batch()
         │  │    └─ Generate embeddings for child chunks only
         │  ├─ [d] Store in SQLite
         │  │    └─ Insert Document, Chunks, update indices
         │  ├─ [e] Store in QdrantDB
         │  │    └─ Upsert embeddings with metadata
         │  ├─ [f] Store in FTS5
         │  │    └─ Insert child chunk content for BM25
         │  └─ [g] Update IngestionRun (progress)
         │
         └─ Final: Mark IngestionRun as completed
         ↓
[4] HTTP 202 Accepted + IngestAccepted response
         ↓
[5] Client polls GET /api/v1/ingest/{run_id} for status
```

### 3. Hybrid Retrieval Workflow

```
Query: "unlicensed driver insurance liability"
         ↓
Retriever.retrieve(query, n=10, search_mode="hybrid")
         ├─ [Dense Path]
         │  └─ embedder.embed_one(query)
         │  └─ QdrantDB.search(query_embedding, limit=20)
         │  └─ Returns: [RankedChunk, ...]
         │
         ├─ [Sparse Path]
         │  └─ SparseRetriever.search(query, limit=20)
         │  └─ Query FTS5 for BM25 ranking
         │  └─ Returns: [RankedChunk, ...]
         │
         └─ [RRF Fusion]
            └─ Merge dense_results + sparse_results
            └─ For each document_id:
            │  └─ RRF_score = 1/(60+rank_dense) + 1/(60+rank_sparse)
            └─ Sort by RRF_score
            └─ Return top-n unique documents
         ↓
Expand parent context
         └─ For each retrieved child chunk:
            └─ Query chunks table for parent_id
            └─ Attach parent_content to RankedChunk
         ↓
Return: [RankedChunk with both child excerpt + parent context, ...]
```

---

## Agent Workflow (5-Node Graph)

### Overview

The `GraphWorkflow` orchestrates 5 specialized nodes for autonomous legal reasoning.

```
┌─────────────────────────────────────────────────────────────────┐
│                        GraphWorkflow.run()                       │
└─────────────────────────────────────────────────────────────────┘
         ↓
    [Conversational Fast Path?]
         ├─ YES → _conversational_response()
         │        └─ Return direct LLM reply (no graph)
         │
         └─ NO → Continue
         ↓
    [1] PlannerNode.execute(state)
         ├─ Input: query_text, recent conversation history
         ├─ LLM Call: PLANNER_PROMPT
         ├─ Output: PlannerOutput
         │  ├─ query_type ('precedent_research', 'general_query', 'conversational')
         │  ├─ depth ('shallow', 'medium', 'deep')
         │  ├─ sub_queries: ['search phrase 1', 'search phrase 2', ...]
         │  └─ legal_issues: ['precise legal question', ...]
         ├─ Fallback: keyword-based heuristic if LLM fails
         └─ Emit event: "thinking:planning"
         ↓
    [Check Query Type]
         ├─ strategy == "conversational"?
         │  └─ YES → _conversational_response()
         │
         ├─ query_type == "general_query"?
         │  └─ YES → [2] RetrievalNode → Skip IRAC → _general_query_response()
         │
         └─ query_type == "precedent_research"?
            └─ YES → Full pipeline [2] → [3] → [4] → [5]
         ↓
    [2] RetrievalNode.execute(state)
         ├─ Input: state.plan.sub_queries
         ├─ For each sub_query:
         │  └─ retriever.retrieve(sub_query, n=12, search_mode="hybrid")
         │  └─ Accumulate results
         ├─ Deduplicate by document_id, keep highest-scoring chunk per doc
         ├─ Limit to _MAX_CONTEXT_CHUNKS = 15
         ├─ Output: state.deduped_context
         └─ Emit event: "thinking:retrieval"
         ↓
    [Check Empty Results]
         ├─ No chunks retrieved?
         │  └─ YES → _no_results_response()
         │
         └─ NO → Continue
         ↓
    [3] ReasonerNode.execute(state)
         ├─ Input: state.deduped_context + state.plan.legal_issues
         ├─ LLM Call: IRAC_REASONING_PROMPT
         ├─ Output: IRACReasoning
         │  ├─ issue: "Precise legal question"
         │  ├─ applicable_rules: ["Rule from Case A", "Rule from Case B", ...]
         │  ├─ application: "Step-by-step application"
         │  ├─ preliminary_conclusion: "Likely outcome"
         │  ├─ precedent_strengths: {doc_id: 0.0–1.0, ...}
         │  └─ contradictions: [{doc_id_a, doc_id_b, description}, ...]
         ├─ Fallback: derive strengths from relevance_score if LLM fails
         ├─ Emit event: "reasoning"
         └─ Output: state.irac
         ↓
    [4] ReflectorNode.execute(state)
         ├─ Input: state.irac + state.deduped_context
         ├─ LLM Call: REFLECTION_PROMPT
         ├─ Output: ReflectionResult
         │  ├─ confidence: 0.0–1.0
         │  ├─ reasoning_quality: 'sufficient', 'needs_improvement', 'insufficient'
         │  ├─ missing_aspects: ["aspect 1", "aspect 2", ...]
         │  ├─ needs_more_retrieval: bool
         │  ├─ refinement_queries: ["refined query 1", ...]
         │  └─ contradictions_addressed: bool
         ├─ Emit event: "thinking:reflection"
         └─ Output: state.reflection
         ↓
    [Loop Decision]
         ├─ confidence < 0.6 AND loop_count < 2 AND refinement_queries exist?
         │  ├─ YES → [Loop back to RetrievalNode]
         │  │        ├─ state.plan.sub_queries = state.reflection.refinement_queries
         │  │        ├─ state.retrieval_iteration += 1
         │  │        └─ Goto [2]
         │  │
         │  └─ NO → Continue
         │
         └─ Proceed to Synthesis
         ↓
    [5] SynthesisNode.execute(state)
         ├─ Resolve mode (research vs general)
         │
         ├─ If RESEARCH mode:
         │  ├─ LLM Call: RESEARCH_SYNTHESIS_PROMPT
         │  ├─ Output: PrecedentAnalysis JSON
         │  │  ├─ supporting_precedents (filtered by strength ≥ 0.5)
         │  │  ├─ adverse_precedents (filtered by strength < 0.5)
         │  │  └─ strategy_recommendation
         │  │
         │  ├─ LLM Stream: RESEARCH_CHAT_SYSTEM_PROMPT
         │  ├─ Output: Streamed narrative response
         │  └─ state.result = {"query_type": "precedent_research", ...}
         │
         └─ If GENERAL mode:
            ├─ LLM Stream: Direct answer + IRAC context
            ├─ Output: Streamed response
            └─ state.result = {"query_type": "general_query", ...}
         ↓
    Return: state.result
         ├─ query_type
         ├─ chat_response (streamed narrative)
         ├─ response (structured JSON)
         ├─ sources_searched (count of deduped docs)
         ├─ processing_time_ms
         └─ agent_steps (all emitted events)
```

### Node Implementation Details

#### PlannerNode

**File**: `src/agent/graph/nodes.py`

**Prompt**: `PLANNER_PROMPT` from `src/agent/prompts.py`

**Decision Logic**:
1. **Conversational check** (exact match): "hi", "hello", "thanks", "bye", "what can you do"
2. **Follow-up references** (context): "which of these", "these judgments", "those cases"
3. **Strategic keywords** (intent): "strategy", "precedent", "adverse", "support our case", "argue"

**Fallback**: If LLM unavailable, use keyword heuristics.

**Output**: `PlannerOutput` with query_type + sub_queries.

#### RetrievalNode

**File**: `src/agent/graph/nodes.py`

**No LLM Calls**: Purely deterministic corpus search.

**Process**:
```python
for sub_query in state.plan.sub_queries:
    results = await retriever.retrieve(sub_query, n=12, search_mode="hybrid")
    state.all_retrieved.extend(results)

# Deduplicate by document_id
deduped = {}
for chunk in state.all_retrieved:
    doc_id = chunk.document_id
    if doc_id not in deduped or chunk.relevance_score > deduped[doc_id].relevance_score:
        deduped[doc_id] = chunk

state.deduped_context = sorted(deduped.values(), key=lambda c: c.relevance_score, reverse=True)[:15]
```

#### ReasonerNode

**File**: `src/agent/graph/nodes.py`

**Prompt**: `IRAC_REASONING_PROMPT` from `src/agent/prompts.py`

**LLM Call**: Formats context + calls LLM once.

**Output**: `IRACReasoning` with precedent_strengths (0–1) per document.

**Strength Scoring**:
- LLM-generated: Based on factual alignment + legal bindingness
- Fallback: Derived from retrieval relevance_score

**Contradictions**: LLM detects when two precedents conflict on the same rule.

#### ReflectorNode

**File**: `src/agent/graph/nodes.py`

**Prompt**: `REFLECTION_PROMPT` from `src/agent/prompts.py`

**Evaluation**:
- **Confidence**: 0.0–1.0 (does the LLM feel confident in the reasoning?)
- **Quality**: 'sufficient', 'needs_improvement', 'insufficient'
- **Gaps**: missing_aspects (what's not covered by current precedents?)
- **Refinement**: refinement_queries (targeted follow-up searches)

**Loop Decision**:
```python
if (state.reflection.confidence < 0.6 and 
    state.retrieval_iteration < 2 and 
    state.reflection.refinement_queries):
    # Loop back to RetrievalNode
    state.plan.sub_queries = state.reflection.refinement_queries
    state.retrieval_iteration += 1
    # Goto RetrievalNode again
```

#### SynthesisNode

**File**: `src/agent/graph/nodes.py`

**Dual Synthesis**:

**Research Mode** (`query_type == "precedent_research"`):
1. LLM calls `RESEARCH_SYNTHESIS_PROMPT` to generate structured `PrecedentAnalysis`
   - Split supporting (strength ≥ 0.5) vs. adverse (< 0.5)
   - Include strategy recommendation
2. LLM streams narrative response grounded in IRAC + JSON results

**General Mode** (`query_type == "general_query"`):
1. LLM streams direct answer
2. No structured JSON output (just chat_response)

**Output**: `state.result` with all final details.

---

## Retrieval System

### Architecture

```
Query: "unlicensed driver insurance liability"
         ↓
    [DenseRetriever]
    └─ Embed query → QdrantDB vector search
    └─ Top-20 results by cosine similarity
    └─ Output: [RankedChunk{rrf_score=0}, ...]
         ↓
    [SparseRetriever]
    └─ Query FTS5 table for BM25 matches
    └─ Top-20 results by BM25 score
    └─ Output: [RankedChunk{rrf_score=0}, ...]
         ↓
    [RRF Fusion]
    └─ For each unique document_id across both lists:
       RRF_score = 1/(60+rank_dense) + 1/(60+rank_sparse)
    └─ Sort by RRF_score (descending)
    └─ Top-10 output
         ↓
    [Parent Expansion]
    └─ For each child chunk:
       ├─ Query chunks table for parent_id
       └─ Attach parent_content to RankedChunk
         ↓
    Output: [RankedChunk with full context, ...]
```

### Components

#### DenseRetriever

**File**: `src/retrieval/dense.py`

**Implementation**:
```python
class DenseRetriever:
    async def search(self, query_embedding: list[float], n_results: int = 10) -> list[RankedChunk]:
        # QdrantDB search
        points = await self.vector_store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=n_results,
            with_payload=True
        )
        return [
            RankedChunk(
                chunk_id=p.id,
                document_id=p.payload['document_id'],
                content=p.payload['content'],
                relevance_score=p.score,
                rrf_score=0  # Set by RRF fusion
            )
            for p in points
        ]
```

#### SparseRetriever

**File**: `src/retrieval/sparse.py`

**Implementation**:
```python
class SparseRetriever:
    async def search(self, query: str, n_results: int = 10) -> list[RankedChunk]:
        # SQLite FTS5 query
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT chunk_id, document_id, content, rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": query, "limit": n_results * 2}
            )
            rows = result.mappings().all()
        
        return [
            RankedChunk(
                chunk_id=row['chunk_id'],
                document_id=row['document_id'],
                content=row['content'],
                relevance_score=abs(row['rank']),  # BM25 rank
                rrf_score=0
            )
            for row in rows
        ]
```

#### RRF Fusion

**File**: `src/retrieval/hybrid.py`

**Formula**:
```
RRF(d) = 1/(60 + rank_dense(d)) + 1/(60 + rank_sparse(d))
```

**Implementation**:
```python
def rrf_fuse(dense_results: list[RankedChunk], 
             sparse_results: list[RankedChunk], 
             limit: int = 10) -> list[RankedChunk]:
    rrf_scores = {}
    
    # Dense results
    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1/(60 + rank)
    
    # Sparse results
    for rank, chunk in enumerate(sparse_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1/(60 + rank)
    
    # Sort and return top-limit
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [chunk for chunk_id, score in sorted_chunks]
```

#### Parent Expansion

**File**: `src/retrieval/retriever.py`

**Purpose**: Retrieve the parent chunk context for LLM reasoning.

**Implementation**:
```python
async def _expand_parent_context(self, chunks: list[RankedChunk]) -> list[RankedChunk]:
    """Attach parent content for LLM reasoning."""
    chunk_ids = [c.chunk_id for c in chunks]
    
    # Query parent chunks
    stmt = text("""
        SELECT c_child.id AS child_id, c_parent.content AS parent_content
        FROM chunks c_child
        JOIN chunks c_parent ON c_child.parent_id = c_parent.id
        WHERE c_child.id IN :ids
    """)
    
    async with self.session_factory() as session:
        result = await session.execute(stmt, {"ids": chunk_ids})
        parent_map = {row['child_id']: row['parent_content'] for row in result}
    
    # Attach parent_content to chunks
    return [
        chunk.model_copy(update={"parent_content": parent_map.get(chunk.chunk_id)})
        for chunk in chunks
    ]
```

---

## Ingestion Pipeline

### End-to-End Process

```
[Input] Corpus Directory (/path/to/pdfs)
         ↓
    [1] Parser Stage
         ├─ For each PDF file:
         │  ├─ read PDF using pdfplumber
         │  ├─ extract text
         │  ├─ extract metadata (case_name, court, date)
         │  ├─ compute file_hash = SHA256(content)
         │  ├─ count pages
         │  └─ Output: ParsedDocument {file_hash, text, case_name, page_count, ...}
         │
         └─ Skip if file already indexed (by hash)
         ↓
    [2] Chunking Stage
         ├─ For each ParsedDocument:
         │  ├─ Detect sections (Facts, Issues, Findings, Judgment, etc.)
         │  ├─ Split each section into parent chunks (~2000 chars, 20% overlap)
         │  ├─ Split each parent into child chunks (~700 chars, 20% overlap)
         │  ├─ Preserve chunk hierarchy (child.parent_id = parent.id)
         │  └─ Output: [ChunkSlice{content, section, chunk_type, parent_index}, ...]
         │
         └─ If no chunks: log error, skip document
         ↓
    [3] Embedding Stage
         ├─ For each child chunk:
         │  └─ Generate embedding using SentenceTransformer
         │  └─ Batch encode in groups of 128 for efficiency
         │  └─ Output: {chunk_id: [0.123, 0.456, ..., 0.789], ...}
         │
         └─ Only child chunks are embedded (not parents)
         ↓
    [4] Storage Stage
         ├─ SQLite:
         │  ├─ Insert Document (id, file_name, case_name, ...)
         │  ├─ Insert parent Chunks
         │  ├─ Insert child Chunks with parent_id reference
         │  └─ Index FTS5: Insert child chunk content for BM25
         │
         ├─ QdrantDB:
         │  ├─ Upsert embeddings with payload {document_id, chunk_index, section, chunk_type}
         │  └─ Commit to collection
         │
         └─ Update IngestionRun (progress counters)
         ↓
[Output] Indexed Corpus
    ├─ SQLite: Document + Chunk records + FTS5 index
    ├─ QdrantDB: Vector points with metadata
    └─ IngestionRun: Completion status
```

### Key Components

#### Parser

**File**: `src/ingestion/parser.py`

**Function**: `parse_pdf(file_path)`

**Returns**: `ParsedDocument`
```python
@dataclass
class ParsedDocument:
    file_hash: str              # SHA256 of content
    file_name: str
    raw_text: str               # Extracted full text
    case_name: str | None
    court_name: str | None
    judgment_date: str | None
    page_count: int
    char_count: int
```

#### Chunker

**File**: `src/ingestion/chunker.py`

**Hierarchical Chunking**:
1. **Section Detection**: Regex patterns identify legal sections
2. **Parent Chunking**: Split each section into ~2000-char blocks with 20% overlap
3. **Child Chunking**: Split each parent into ~700-char units with 20% overlap

**Regex Patterns**:
```
Sections: FACTS, ISSUES, ARGUMENTS, FINDINGS, JUDGMENT, etc.
```

**Output**: `list[ChunkSlice]`
```python
@dataclass
class ChunkSlice:
    content: str
    char_start: int
    char_end: int
    section: str                # 'facts', 'issues', 'findings', etc.
    chunk_type: str             # 'parent', 'child'
    parent_index: int | None    # If child, index into parent sublist
```

#### Embedder

**File**: `src/ingestion/embedder.py`

**Model**: `all-MiniLM-L6-v2` (384-dim embeddings)

**Implementation**:
```python
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def embed_one(self, text: str) -> list[float]:
        # Embed single text
        embedding = await asyncio.get_event_loop().run_in_executor(
            None, self.model.encode, text
        )
        return embedding.tolist()
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Embed batch in parallel
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, self.model.encode, texts
        )
        return embeddings.tolist()
```

#### Ingestion Pipeline Orchestrator

**File**: `src/ingestion/pipeline.py`

**Class**: `IngestionPipeline`

**Process**:
```python
async def run(self, corpus_dir: str, run_id: str) -> None:
    files = sorted(Path(corpus_dir).glob("*.pdf"))
    
    for file_path in files:
        try:
            # [1] Parse
            parsed = await parse_pdf(file_path)
            
            # [2] Check if already indexed
            existing = await self.document_repository.get_by_filename(file_name)
            if existing and already indexed:
                continue
            
            # [3] Chunk
            chunk_slices = await self.chunker.chunk_text(parsed.raw_text)
            parent_slices = [s for s in chunk_slices if s.chunk_type == "parent"]
            child_slices = [s for s in chunk_slices if s.chunk_type == "child"]
            
            # [4] Embed (only children)
            embeddings = await self.embedder.embed_batch([s.content for s in child_slices])
            
            # [5] Store
            await self.document_repository.insert(document)
            await self.chunk_repository.insert_batch(chunks)
            await self.vector_store.upsert(embeddings_with_metadata)
            
        except Exception as e:
            await self.ingestion_failure_repository.insert(IngestionFailureRecord(...))
```

---

## LLM Integration

### LLM Provider Protocol

**File**: `src/llm/base.py`

**Protocol Definition**:
```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]: ...

@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]
```

### OpenAI Adapter

**File**: `src/llm/openai_adapter.py`

**Implementation**: Wraps OpenAI Python client to support:
- **OpenAI API** (default)
- **Groq API** (via `base_url` override)
- **Azure OpenAI** (via `base_url` override)

**Constructor**:
```python
class OpenAIAdapter:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        request_timeout: float = 60.0,
        max_retries: int = 5,
    ): ...
```

**Key Methods**:

#### `chat()` - Synchronous Request

```python
async def chat(
    self,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> LLMResponse:
    """Execute a single LLM call (no streaming)."""
    response = await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        tools=tools,
        temperature=0.2,  # Lower temp for reasoning consistency
        timeout=self.request_timeout,
    )
    return LLMResponse(
        content=response.choices[0].message.content,
        tool_calls=[...],  # Parsed function calls
        raw=response.model_dump()
    )
```

#### `chat_stream()` - Streaming Response

```python
async def chat_stream(
    self,
    messages: list[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Stream response tokens as they arrive."""
    async with await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        stream=True,
    ) as stream:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### LLM Factory

**File**: `src/llm/factory.py`

**Purpose**: Create LLMProvider instance from configuration.

**Implementation**:
```python
class LLMFactory:
    @staticmethod
    def from_config(settings: Settings) -> OpenAIAdapter:
        provider = settings.llm_provider.lower()
        
        if provider == "openai":
            base_url = settings.llm_base_url
        elif provider == "groq":
            base_url = settings.llm_base_url or "https://api.groq.com/openai/v1"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return OpenAIAdapter(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=base_url,
            request_timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
        )
```

### Configuration

**Environment Variables**:
```bash
LLM_PROVIDER=openai              # or 'groq', 'azure'
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1  # Optional override
LLM_REQUEST_TIMEOUT=60.0
LLM_MAX_RETRIES=5
```

---

## Error Handling

### Exception Hierarchy

**File**: `src/core/exceptions.py`

```
CaseyError (base)
├─ CorpusNotIndexedError         (HTTP 409)
├─ IngestionError                (HTTP 400)
├─ LLMUnavailableError           (HTTP 503)
├─ RetrievalError                (HTTP 500)
└─ ValidationError               (HTTP 400)
```

### Error Response Format

All errors return JSON:
```json
{
  "correlation_id": "uuid-xxx",
  "error_code": "CORPUS_NOT_INDEXED",
  "message": "Corpus has not been indexed yet. Please trigger ingestion.",
  "timestamp": "2026-04-28T10:00:00Z"
}
```

### Exception Handlers

**File**: `src/main.py`

```python
@app.exception_handler(CorpusNotIndexedError)
async def handle_corpus_not_indexed(request, exc):
    return JSONResponse(
        status_code=409,
        content={"error_code": "CORPUS_NOT_INDEXED", ...}
    )
```

### Common Error Scenarios

| Scenario | Exception | HTTP Status | Action |
|----------|-----------|-------------|--------|
| Corpus empty on query | CorpusNotIndexedError | 409 | Suggest ingestion |
| LLM API timeout | LLMUnavailableError | 503 | Retry later |
| Invalid PDF | IngestionError | 400 | Log and skip file |
| Invalid query payload | ValidationError | 400 | Return validation errors |
| Database connectivity | RetrievalError | 500 | Log, retry with exponential backoff |

---

## Configuration & Runtime

### Settings

**File**: `src/core/config.py`

**Class**: `Settings` (Pydantic BaseSettings)

**Environment Variables**:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` | LLM provider (openai, groq) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_API_KEY` | (required) | API key |
| `CORPUS_DIR` | `judgement_pdfs` | PDF corpus directory |
| `SQLITE_DB_PATH` | `data/Casey.db` | SQLite database file |
| `QDRANT_URL` | (optional) | QdrantDB remote URL (leave empty for embedded) |
| `QDRANT_PATH` | `data/qdrant` | QdrantDB embedded storage directory |
| `QDRANT_COLLECTION` | `judgments` | Vector collection name |
| `REDIS_URL` | (optional) | Redis URL for caching (leave empty to disable) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json, text) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

### Runtime Services

**File**: `src/core/runtime.py`

**Class**: `RuntimeServices` (dataclass with all singletons)

**Lazy Initialization**:
```python
async def ensure_runtime(app: FastAPI) -> RuntimeServices:
    """Initialize all runtime services on first access."""
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        return runtime  # Return cached
    
    # Initialize (only once, protected by lock)
    async with app.state.runtime_lock:
        # Create all components...
        runtime = RuntimeServices(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            vector_store=vector_store,
            cache=cache,
            # ... all repositories, services, embedder, chunker, llm_provider
        )
        app.state.runtime = runtime
        return runtime
```

### Logging

**File**: `src/core/logging.py`

**Features**:
- Structured JSON logging with `structlog`
- Correlation ID tracking (request-scoped)
- Component-level logging (e.g., `component="agent"`)

**Example**:
```python
logger.info(
    "query.executed",
    correlation_id=correlation_id,
    query_type=result["query_type"],
    processing_time_ms=result["processing_time_ms"],
    sources_searched=result["sources_searched"],
)
```

### Caching

**File**: `src/core/cache.py`

**Purpose**: Optional Redis caching for:
- Document metadata
- Chat history
- Chunk lists
- Document/chunk counts

**TTL Values**:
```python
TTL_DOC = 3600              # 1 hour
TTL_DOC_LIST = 1800        # 30 minutes
TTL_DOC_COUNT = 1800       # 30 minutes
TTL_CHAT_HISTORY = 86400   # 1 day
TTL_CHUNK_LIST = 1800      # 30 minutes
TTL_CHUNK_COUNT = 1800     # 30 minutes
TTL_CHAT_RECENT = 86400    # 1 day
```

**Cache Miss Behavior**: If Redis unavailable, system continues without caching (graceful degradation).

---

## Deployment & Lifecycle

### Startup Sequence

```
[1] FastAPI.lifespan context manager enters
    └─ configure_logging()
    └─ _log_startup_configuration()
         
[2] ensure_runtime() is called
    ├─ Create SQLite connection pool
    ├─ Initialize database schema (create tables if not exist)
    ├─ Create QdrantDB client (embedded or remote)
    ├─ Initialize QdrantDB collection
    ├─ Connect to Redis (if configured)
    ├─ Create repositories
    ├─ Initialize embedder (SentenceTransformer model download)
    ├─ Initialize chunker
    ├─ Create LLM provider (validate API key)
    └─ Create all services
         
[3] Server listens on HOST:PORT
    ├─ /health endpoint available
    └─ All API routes ready
```

### Shutdown Sequence

```
[1] SIGTERM/SIGINT received (Ctrl+C or pod termination)
    
[2] close_runtime() called
    ├─ Close SQLite connection pool
    ├─ Disconnect Redis (if connected)
    ├─ Close QdrantDB connection (if remote)
    └─ Cleanup temporary resources
    
[3] FastAPI.lifespan context manager exits
    ├─ logger.info("application.shutdown")
    
[4] Server stops
```

### Docker Deployment

**Dockerfile** example:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment Setup**:
```bash
# .env file
LLM_API_KEY=sk-...
CORPUS_DIR=/data/pdfs
SQLITE_DB_PATH=/data/casey.db
QDRANT_PATH=/data/qdrant
```

### Performance Tuning

| Parameter | Default | Recommendation | Reason |
|-----------|---------|---|---------|
| `_MAX_REFLECTION_LOOPS` | 2 | 1–2 | Control query latency |
| `_MAX_CONTEXT_CHUNKS` | 15 | 10–20 | Balance LLM token limit |
| `_INTER_REQUEST_DELAY` | 1.0 sec | 0.5–1.0 | Groq rate-limit (150 req/min) |
| Embedder batch size | 128 | 64–256 | Memory vs. throughput |
| SQLite WAL mode | Enabled | Keep enabled | Better concurrency |
| QdrantDB vector index | HNSW | Default | Fast approximate NN search |

---

## Key Workflows Summary

### 1. User Query Execution

```
User Query → API Route → QueryService → LegalResearchAgent 
→ GraphWorkflow (5 nodes) → LLM calls + Retrieval 
→ Chat history storage → Response JSON/Stream
```

### 2. PDF Ingestion

```
PDF Files → Parser → Chunker (hierarchical) → Embedder 
→ Storage (SQLite, QdrantDB, FTS5) → IngestionRun tracking
```

### 3. Corpus Search (Retrieval)

```
Query → Embedder (dense) + FTS5 (sparse) → RRF Fusion 
→ Parent Expansion → Ranked Results
```

### 4. Agent Reasoning (5-node graph)

```
Planner → Retrieval → Reasoner (IRAC) → Reflector (confidence) 
→ [Loop?] → Synthesis → Final Response
```

---

## Debugging Tips

### Enable Detailed Logging

```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
python -m uvicorn src.main:app
```

### Check Corpus Status

```bash
curl http://localhost:8000/api/v1/documents
```

### Trigger Ingestion

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus_dir": "/path/to/pdfs"}'
```

### Check Ingestion Progress

```bash
curl http://localhost:8000/api/v1/ingest/{run_id}
```

### Execute Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Build a case for unlicensed driver insurance claim"}'
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Agent** | LLM-powered autonomous legal research system |
| **Agent Step** | Single emission from a graph node (thinking, reasoning, synthesis, etc.) |
| **Chunk** | Segment of document text (parent or child) |
| **Correlation ID** | Request-scoped UUID for tracing |
| **Dense Retrieval** | Vector similarity search (QdrantDB) |
| **Hybrid Retrieval** | Combination of dense + sparse with RRF fusion |
| **IRAC** | Issue, Rules, Application, Conclusion (legal reasoning framework) |
| **Node** | Stage in the 5-node graph workflow |
| **Parent Chunk** | Large context block (~2000 chars) for LLM reasoning |
| **Child Chunk** | Small retrieval unit (~700 chars) embedded for search |
| **Precedent Strength** | Score 0–1 indicating legal bindingness of a judgment |
| **RRF** | Reciprocal Rank Fusion (merge dense + sparse scores) |
| **Runtime** | Lazy-initialized singleton services |
| **Sparse Retrieval** | BM25 keyword search (SQLite FTS5) |

---

## Evaluation System

### Overview

The **Evaluation System** measures the quality of the Legal Research Agent across four critical dimensions:

1. **Precision** — Are cited precedents actually relevant to the query?
2. **Recall** — Does the agent find all (or most) important precedents?
3. **Reasoning** — Is the legal explanation correct, detailed, and free of hallucinations?
4. **Adverse** — Does the agent honestly surface unfavourable precedents?

**Key Design**: Hybrid Rule-Based + LLM-as-Judge
- **Rule-based scoring** (precision/recall): When ground-truth document IDs exist, deterministic calculation
- **LLM judge**: Always evaluates reasoning quality; fallback for precision/recall when no ground truth available
- **Hard overrides**: Adverse precedent failure is always scored 0.0 if the agent returns an empty adverse list

### Architecture

```
evals/
├── runner.py           # Main entry point: orchestrates bootstrap + case execution
├── evaluator.py        # LegalAgentEvaluator: runs 4-dimension scoring
├── benchmark.py        # BenchmarkCase definitions (4 test cases with ground truth)
├── schemas.py          # Pydantic models (EvaluationInput, EvaluationResult)
├── prompts.py          # LLM judge system + user prompts
├── EVAL_FLOW.md        # This file
├── results/            # Output directory (JSON reports)
└── __init__.py
```

### Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    python -m evals.runner                        │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ [1] Bootstrap                                                    │
│  ├─ Load settings from .env (same as production)               │
│  ├─ Initialize SQLite database + schema                        │
│  ├─ Initialize QdrantDB (uses temporary isolated directory)    │
│  ├─ Ingest corpus from judgement_pdfs/                         │
│  ├─ Create LegalResearchAgent + LegalAgentEvaluator            │
│  └─ Share single LLM provider across both                      │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ [2] For Each BenchmarkCase in ALL_CASES                         │
│  ├─ Execute: agent.run(case.query, correlation_id)            │
│  ├─ Normalize response (PrecedentAnalysis or GeneralQueryResponse) │
│  ├─ Extract ground-truth doc IDs from repository              │
│  └─ Build EvaluationInput                                      │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ [3] Score Each Dimension                                        │
│  ├─ Precision:   Rule-based (if GT) else LLM judge            │
│  ├─ Recall:      Rule-based (if GT) else LLM judge            │
│  ├─ Reasoning:   Always LLM judge                             │
│  └─ Adverse:     Hard override if empty, else LLM judge       │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ [4] Calculate Overall Score & Verdict                           │
│  ├─ overall_score = (precision + recall + reasoning + adverse) / 4 │
│  ├─ final_verdict: "poor" | "average" | "good" | "excellent" │
│  └─ Thresholds:                                                 │
│     ├─ excellent ≥ 0.80                                        │
│     ├─ good ≥ 0.65                                             │
│     ├─ average ≥ 0.45                                          │
│     └─ poor < 0.45                                             │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ [5] Write Report & Print Summary                                │
│  ├─ Save JSON report to evals/results/report_<timestamp>.json │
│  ├─ Print human-readable summary table                         │
│  ├─ Calculate pass rate (vs. per-case minimum thresholds)     │
│  └─ Perform failure analysis                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Evaluation Dimensions

#### 1. Precision

**Metric**: `relevant_cited / total_cited`

**Rule-Based Calculation** (when ground truth provided):
```python
relevant = [doc_id for doc_id in cited_ids if doc_id in ground_truth_set]
precision = len(relevant) / len(cited_ids)
```

**LLM Judge** (fallback when no ground truth):
- Asks LLM: "How many of the cited cases are actually relevant to the user's query?"
- Considers degree of relevance (strong vs. weak connection)
- Returns: score (0–1) + relevant_cases_count + total_cases_cited

**Output**:
```python
class PrecisionResult(BaseModel):
    score: float                    # 0.0–1.0
    relevant_cases_count: int       # Number of relevant cases
    total_cases_cited: int          # All cases cited
    explanation: str                # Reasoning
```

#### 2. Recall

**Metric**: `ground_truth_found / total_ground_truth`

**Rule-Based Calculation** (when ground truth provided):
```python
gt_set = set(ground_truth_doc_ids)
cited_set = set(agent_cited_ids)
found = gt_set & cited_set
recall = len(found) / len(gt_set)
missed = sorted(gt_set - cited_set)
```

**LLM Judge** (fallback when no ground truth):
- Asks: "Are there obvious or important precedents missing from the agent's response?"
- Infers from expected_themes in BenchmarkCase
- Returns: score (0–1) + missed_key_precedents list

**Output**:
```python
class RecallResult(BaseModel):
    score: float                    # 0.0–1.0
    missed_key_precedents: list[str]  # Document IDs or descriptions
    explanation: str
```

#### 3. Reasoning Quality

**Metric**: Legal explanation depth, correctness, and coherence

**Always LLM Judge** (never rule-based):
- Evaluates whether the agent explains:
  - Why each precedent applies (or does not apply)
  - The legal principles involved
  - Factual alignment between precedent and current case
  - Distinguishing factors when adverse
- Penalizes shallow reasoning, vague statements, contradictions, and hallucinations

**Output**:
```python
class ReasoningResult(BaseModel):
    score: float                    # 0.0–1.0
    strengths: list[str]            # What was well-explained
    weaknesses: list[str]           # What was poorly explained
    hallucinations: list[str]       # Factual errors or made-up cases
```

#### 4. Adverse Precedent Identification

**Metric**: Honest identification of unfavourable precedents

**Hard Override Rule**:
```python
if len(agent_response.get("adverse_precedents", [])) == 0:
    score = 0.0
    verdict = "CRITICAL FAILURE: Agent returned ZERO adverse precedents"
```

**LLM Judge** (if adverse list is not empty):
- Evaluates quality of adverse identification
- Checks whether risks are clearly explained and not downplayed
- Returns: score (0–1) + adverse_cases_identified + missing_adverse_cases

**Output**:
```python
class AdverseResult(BaseModel):
    score: float                    # 0.0–1.0
    adverse_cases_identified: list[str]  # Cases agent found
    missing_adverse_cases: list[str]     # Risks agent missed
    risk_analysis_quality: str      # "excellent", "adequate", "poor", etc.
```

### Benchmark Cases

All benchmark cases are defined in [evals/benchmark.py](evals/benchmark.py) with:
- **case_id**: Unique identifier
- **query**: Full legal query (simulating real user input)
- **expected_themes**: Keywords/concepts that must be addressed (for recall inference)
- **expected_adverse_themes**: Adverse concepts to check
- **min_precision, min_recall, min_reasoning, min_adverse**: Pass/fail thresholds

#### Case 1: Motor Accident, Unlicensed Driver, Insurer Liability

**case_id**: `motor_accident_001`

**Query**: Mrs. Lakshmi Devi injured by commercial truck (driver unlicensed). Insurer denies liability citing policy exclusion. Seeking ₹35L compensation.

**Expected Themes**:
- insurer liability unlicensed driver
- pay and recover doctrine
- third party motor accident claim
- MACT compensation permanent disability
- contributory negligence

**Expected Adverse Themes**:
- policy exclusion breach condition
- insurer not liable gratuitous passenger
- no compensation rash negligent victim

**Min Thresholds**:
- Precision: 0.55
- Recall: 0.50
- Reasoning: 0.55
- Adverse: 0.50

#### Case 2: Fatal Accident, KSRTC Bus, Compensation Quantum

**case_id**: `fatal_accident_002`

**Query**: Mr. Rajesh Sharma (₹25K/month) killed by KSRTC bus. Wife + 2 minor children claiming. Bus driver was rash.

**Expected Themes**:
- multiplier method Sarla Verma
- notional income deceased non-earning
- loss of dependency calculation
- future prospects addition income
- KSRTC government corporation liability

**Expected Adverse Themes**:
- contributory negligence pedestrian award reduced
- no proof actual income compensation nominal

**Min Thresholds**:
- Precision: 0.55
- Recall: 0.45
- Reasoning: 0.50
- Adverse: 0.45

#### Case 3: General Query (Non-Precedent Research)

**case_id**: `general_query_003`

**Query**: "What is the Motor Vehicles Act 1988?" (Exploratory, no adversity expected)

**Expected Themes**:
- Motor Vehicles Act structure
- MACT jurisdiction
- compensation framework

**Expected Adverse Themes**: (empty list)

**Min Thresholds**: Lower thresholds (system expects less precision on encyclopedic queries)

#### Case 4: Adverse-Heavy Query

**case_id**: `adverse_heavy_004`

**Query**: Intoxicated driver claim query (must surface strong adverse precedents)

**Expected Themes**:
- intoxication recklessness
- insurer liability exclusion
- policy breach material fact

**Expected Adverse Themes**:
- intoxication insurer not liable
- rash negligence no compensation
- policy exclusion applies

**Min Thresholds**:
- Adverse: 0.60 (higher minimum due to critical importance)

### Implementation Details

#### File: `evals/runner.py`

**Main Entry Point**: `async def main()`

**Key Functions**:

1. **`_bootstrap()` → (agent, evaluator, doc_repository, engine)**
   - Initialize all production components (SQLite, QdrantDB, LLM provider)
   - Ingest corpus if not already present
   - Return ready-to-use agent + evaluator

2. **`_run_case(case, agent, evaluator, doc_repository) → dict`**
   - Execute agent on one benchmark case
   - Normalize response (handle both PrecedentAnalysis and GeneralQueryResponse)
   - Call evaluator.evaluate()
   - Validate pass/fail against min thresholds
   - Print per-case results

3. **`_write_report(case_results) → Path`**
   - Save all results to JSON file with timestamp
   - Calculate summary statistics (pass rate, avg score)
   - Build failure analysis
   - Write to `evals/results/report_<timestamp>.json`

**Usage**:
```bash
cd /path/to/backend
export LLM_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
python -m evals.runner
```

#### File: `evals/evaluator.py`

**Class**: `LegalAgentEvaluator`

**Constructor**:
```python
class LegalAgentEvaluator:
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
```

**Main Method**: `async def evaluate(eval_input: EvaluationInput) → EvaluationResult`

**Process**:
```python
async def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
    cited_ids = _extract_cited_ids(eval_input.agent_response)
    
    # Precision
    if eval_input.ground_truth_docs:
        precision = _rule_precision(cited_ids, eval_input.ground_truth_docs)
    else:
        precision = await self._llm_precision(eval_input)
    
    # Recall
    if eval_input.ground_truth_docs:
        recall = _rule_recall(cited_ids, eval_input.ground_truth_docs)
    else:
        recall = await self._llm_recall(eval_input)
    
    # Reasoning (always LLM)
    reasoning = await self._llm_reasoning(eval_input)
    
    # Adverse (override if empty, else LLM)
    adverse = _rule_adverse(eval_input.agent_response)
    if adverse is None:
        adverse = await self._llm_adverse(eval_input)
    
    # Overall score (equal weight)
    overall = (precision.score + recall.score + reasoning.score + adverse.score) / 4
    
    return EvaluationResult(
        precision=precision,
        recall=recall,
        reasoning=reasoning,
        adverse=adverse,
        overall_score=round(overall, 3),
        final_verdict=_verdict(overall)
    )
```

#### File: `evals/schemas.py`

**Input Model**:
```python
class EvaluationInput(BaseModel):
    query: str                                          # User's legal query
    retrieved_docs: list[dict[str, Any]]               # Top-K documents (optional)
    agent_response: dict[str, Any]                     # PrecedentAnalysis or GeneralQueryResponse
    ground_truth_docs: list[str] | None = None         # Document IDs (runtime-populated)
```

**Output Model**:
```python
class EvaluationResult(BaseModel):
    precision: PrecisionResult
    recall: RecallResult
    reasoning: ReasoningResult
    adverse: AdverseResult
    overall_score: float                               # 0.0–1.0
    final_verdict: Literal["poor", "average", "good", "excellent"]
```

#### File: `evals/prompts.py`

**JUDGE_SYSTEM_PROMPT**: Sets context for LLM judge
- Explains 4 dimensions
- Requests strict, objective evaluation
- Demands JSON-only output

**JUDGE_USER_TEMPLATE**: Template with placeholders:
- {query}
- {retrieved_docs}
- {ground_truth_docs}
- {agent_response}

**Example LLM Judge Call**:
```python
messages = [
    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
    {"role": "user", "content": JUDGE_USER_TEMPLATE.format(...)}
]
response = await self.llm.chat(messages)
parsed = _parse_judge_response(response.content)  # Extract JSON
```

### Scoring Rules

#### Overall Score Calculation

```python
overall_score = (precision + recall + reasoning + adverse) / 4
```

**Equal weight**: 25% per dimension

#### Verdict Classification

| Score Range | Verdict |
|-------------|---------|
| ≥ 0.80 | excellent |
| ≥ 0.65 < 0.80 | good |
| ≥ 0.45 < 0.65 | average |
| < 0.45 | poor |

#### Pass/Fail Per Case

```python
passed = (
    precision.score >= case.min_precision
    and recall.score >= case.min_recall
    and reasoning.score >= case.min_reasoning
    and adverse.score >= case.min_adverse
)
```

**Case fails if ANY dimension is below minimum.**

### Results & Reporting

#### Report Structure

**File**: `evals/results/report_<YYYYMMDDTHHMMSSZ>.json`

**Schema**:
```json
{
  "generated_at": "2026-04-28T10:15:30.123456+00:00",
  "summary": {
    "total_cases": 4,
    "passed": 3,
    "failed": 1,
    "pass_rate": 0.75,
    "avg_overall_score": 0.72
  },
  "cases": [
    {
      "case_id": "motor_accident_001",
      "query_type": "precedent_research",
      "passed": true,
      "result": {
        "precision": {...},
        "recall": {...},
        "reasoning": {...},
        "adverse": {...},
        "overall_score": 0.75,
        "final_verdict": "good"
      },
      "agent_steps": [...],
      "processing_time_ms": 45000
    },
    ...
  ],
  "failure_analysis": {
    "motor_accident_001": {
      "reasoning": "Missed discussion of pay-and-recover doctrine"
    },
    ...
  }
}
```

#### Console Output Example

```
============================================================
CASE: motor_accident_001
============================================================
Query: Mrs. Lakshmi Devi was severely injured when...

Verdict: GOOD  [PASS]
  Overall:    0.75
  Precision:  0.80  (min 0.55)
  Recall:     0.70  (min 0.50)
  Reasoning:  0.75  (min 0.55)
  Adverse:    0.65  (min 0.50)
  [WARNING] Hallucinations: ["Dharamshala judgment (non-existent case)"]

============================================================
CASE: fatal_accident_002
============================================================
...
```

#### Failure Analysis

If a case fails, the report includes brief analysis:
```python
failure_analysis = {
    "case_id": {
        "precision": "Cited 5 irrelevant cases (bankruptcy law)",
        "reasoning": "Incorrectly applied Sarla Verma multiplier",
        "adverse": None  # Only if this dimension failed
    }
}
```

### Running Evaluations

#### Prerequisites

1. **Environment Variables**:
   ```bash
   export LLM_API_KEY=sk-...
   export LLM_PROVIDER=openai           # or 'groq'
   export LLM_MODEL=gpt-4o-mini
   ```

2. **Corpus Ingested**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/ingest \
     -H "Content-Type: application/json" \
     -d '{"corpus_dir": "judgement_pdfs"}'
   ```

#### Command

```bash
cd /path/to/backend
python -m evals.runner
```

#### What Happens

1. Bootstraps all infrastructure (DB, vectors, LLM)
2. Ingests corpus (if first run)
3. Executes 4 benchmark cases
4. Evaluates each using hybrid rule-based + LLM judge
5. Writes JSON report to `evals/results/`
6. Prints summary to console

#### Output

```
EVALUATION COMPLETE
===================
Total Cases: 4
Passed: 3
Failed: 1
Pass Rate: 0.75 (75%)
Avg Overall Score: 0.72

Report saved to: evals/results/report_20260428T101530Z.json
```

### Key Design Decisions

1. **Hybrid Rule-Based + LLM Judge**
   - Rule-based is deterministic when ground truth exists
   - LLM judge provides nuance and handles missing ground truth
   - Eliminates "LLM judge bias" by grounding in facts

2. **Hard Override on Adverse**
   - An empty adverse list is an automatic failure (score 0.0)
   - Reflects critical legal practice requirement: never hide risks

3. **Isolated QdrantDB per Run**
   - Each evaluation run gets temporary Qdrant directory
   - Prevents lock conflicts with production instances
   - Ensures clean eval state

4. **Shared LLM Provider**
   - Single LLM provider for both agent and evaluator
   - Reduces API call overhead
   - Ensures consistent model across agent + judge

5. **Benchmark Grounded in Real Queries**
   - All cases drawn from Indian motor accident domain (primary use case)
   - Ground truth themes from legal literature + domain expertise
   - Min thresholds set conservatively (expecting 45%+ on baseline)

### Debugging Tips

#### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
python -m evals.runner
```

#### Run Single Case

To evaluate a single case, modify [evals/benchmark.py](evals/benchmark.py):
```python
ALL_CASES = [MOTOR_ACCIDENT_INSURANCE]  # Comment others
```

#### Check Ground Truth Population

The runner populates ground_truth_doc_ids at runtime by querying the document repository:
```python
for case in ALL_CASES:
    if case.ground_truth_file_patterns:
        case.ground_truth_doc_ids = [
            doc.id for doc in docs 
            if any(pattern in doc.file_name for pattern in case.ground_truth_file_patterns)
        ]
```

#### Inspect LLM Judge Prompts

Set breakpoint in `evals/evaluator.py` method `_llm_reasoning()` to see full prompt sent to judge.

---

**End of Codebase Guide**
