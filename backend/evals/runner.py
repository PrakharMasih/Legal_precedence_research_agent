#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation runner — runs the full benchmark suite against the live agent.

Usage (from the repo root, with the virtualenv active):

    python -m evals.runner

Environment variables required (same as the application):
    LLM_API_KEY, LLM_PROVIDER, LLM_MODEL (see .env or src/core/config.py)

The runner:
  1. Boots the same database + vector-store that the application uses.
  2. Runs the agent on every BenchmarkCase in evals/benchmark.py.
  3. Evaluates each response with LegalAgentEvaluator (LLM-as-judge).
  4. Writes a JSON report to evals/results/report_<timestamp>.json.
  5. Prints a human-readable summary table to stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evals.benchmark import ALL_CASES
from evals.evaluator import LegalAgentEvaluator
from evals.schemas import BenchmarkCase, EvaluationInput, EvaluationResult
from src.agent.agent import LegalResearchAgent
from src.agent.tools import ResearchToolbox
from src.core.config import get_settings
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.llm.factory import LLMFactory
from src.retrieval.retriever import Retriever
from src.storage.database import create_db_engine, init_schema, make_session_factory
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionFailureRepository,
    IngestionRunRepository,
)
from src.storage.vector_store import VectorStore

_logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
# Use a temporary directory for Qdrant during evals to avoid lock conflicts
_TEMP_QDRANT_DIR = None


# Infrastructure bootstrap (mirrors src/core/runtime.py but eval-scoped)


async def _bootstrap():
    """Return (agent, evaluator, doc_repository) using live settings but with isolated Qdrant."""
    global _TEMP_QDRANT_DIR
    settings = get_settings()
    engine = create_db_engine(settings.sqlite_db_path)
    await init_schema(engine)
    session_factory = make_session_factory(engine)

    # Use a temporary directory for Qdrant during this eval run
    # This avoids lock conflicts with production/test instances
    if _TEMP_QDRANT_DIR is None:
        _TEMP_QDRANT_DIR = tempfile.TemporaryDirectory(prefix="qdrant_eval_")

    temp_qdrant_path = Path(_TEMP_QDRANT_DIR.name)

    vector_store = VectorStore(
        url=settings.qdrant_url or None,
        api_key=settings.qdrant_api_key or None,
        path=None if settings.qdrant_url else temp_qdrant_path,
        collection=settings.qdrant_collection,
    )
    await vector_store.init_collection()

    document_repository = DocumentRepository(session_factory)
    chunk_repository = ChunkRepository(session_factory)
    ingestion_run_repository = IngestionRunRepository(session_factory)
    ingestion_failure_repository = IngestionFailureRepository(session_factory)

    # Ingest corpus from judgement_pdfs directory
    corpus_dir = Path("judgement_pdfs")
    if corpus_dir.exists():
        ingestion_pipeline = IngestionPipeline(
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            ingestion_run_repository=ingestion_run_repository,
            ingestion_failure_repository=ingestion_failure_repository,
            vector_store=vector_store,
            embedder=Embedder(),
            chunker=Chunker(),
        )
        run_id = str(uuid4())
        _logger.info("Ingesting corpus from %s", corpus_dir)
        await ingestion_pipeline.run(str(corpus_dir), run_id)
        _logger.info("Corpus ingestion complete (run_id=%s)", run_id)

    retriever = Retriever(
        session_factory=session_factory,
        vector_store=vector_store,
        embedder=Embedder(),
    )

    toolbox = ResearchToolbox(
        retriever=retriever,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
    )

    llm_provider = LLMFactory.from_config(settings)

    agent = LegalResearchAgent(llm_provider=llm_provider, toolbox=toolbox)
    evaluator = LegalAgentEvaluator(llm_provider=llm_provider)

    return agent, evaluator, document_repository, engine


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


async def _run_case(
    case: BenchmarkCase,
    agent: LegalResearchAgent,
    evaluator: LegalAgentEvaluator,
    doc_repository: DocumentRepository,
) -> dict:
    """Run one benchmark case and return a serialisable result dict."""
    print(f"\n{'=' * 60}")
    print(f"CASE: {case.case_id}")
    print(f"{'=' * 60}")
    print(f"Query: {case.query[:120]}...")

    correlation_id = str(uuid4())
    agent_result = await agent.run(case.query, correlation_id)

    agent_response: dict = {}
    query_type = agent_result.get("query_type", "")
    raw_response = agent_result.get("response", {})

    # Normalise to dict regardless of whether it's a Pydantic model or plain dict
    if hasattr(raw_response, "model_dump"):
        agent_response = raw_response.model_dump()
    elif isinstance(raw_response, dict):
        agent_response = raw_response
    else:
        agent_response = {}

    # Retrieve ground-truth document IDs from the corpus if the case specifies file patterns
    # (ground_truth_doc_ids in the benchmark are empty by default — populated here at runtime)
    ground_truth = list(case.ground_truth_doc_ids) if case.ground_truth_doc_ids else None

    # For general queries, build a synthetic agent_response shape that the evaluator understands
    if query_type == "general_query":
        docs = agent_response.get("supporting_documents", [])
        # Wrap in a research-like shape so evaluator can extract cited IDs uniformly
        agent_response = {
            "supporting_precedents": [
                {
                    "document_id": d.get("document_id", ""),
                    "file_name": d.get("file_name", ""),
                    "legal_principle": d.get("relevance_score", ""),
                    "factual_alignment": "",
                }
                for d in docs
            ],
            "adverse_precedents": [],
        }

    eval_input = EvaluationInput(
        query=case.query,
        retrieved_docs=[],  # runner does not re-expose raw chunks; evaluator uses agent output
        agent_response=agent_response,
        ground_truth_docs=ground_truth,
    )

    result: EvaluationResult = await evaluator.evaluate(eval_input)

    # Pass/fail assertions
    passed = (
        result.precision.score >= case.min_precision
        and result.recall.score >= case.min_recall
        and result.reasoning.score >= case.min_reasoning
        and result.adverse.score >= case.min_adverse
    )

    _print_result(case, result, passed)

    return {
        "case_id": case.case_id,
        "query_type": query_type,
        "passed": passed,
        "result": result.model_dump(),
        "agent_steps": agent_result.get("agent_steps", []),
        "processing_time_ms": agent_result.get("processing_time_ms", 0),
    }


def _print_result(case: BenchmarkCase, result: EvaluationResult, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\nVerdict: {result.final_verdict.upper()}  [{status}]")
    print(f"  Overall:    {result.overall_score:.2f}")
    print(f"  Precision:  {result.precision.score:.2f}  (min {case.min_precision})")
    print(f"  Recall:     {result.recall.score:.2f}  (min {case.min_recall})")
    print(f"  Reasoning:  {result.reasoning.score:.2f}  (min {case.min_reasoning})")
    print(f"  Adverse:    {result.adverse.score:.2f}  (min {case.min_adverse})")
    if result.reasoning.hallucinations:
        print(f"  [WARNING] Hallucinations: {result.reasoning.hallucinations}")
    if result.recall.missed_key_precedents:
        print(f"  [WARNING] Missed precedents: {result.recall.missed_key_precedents}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(case_results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS_DIR / f"report_{timestamp}.json"

    total = len(case_results)
    passed = sum(1 for r in case_results if r["passed"])
    avg_overall = sum(r["result"]["overall_score"] for r in case_results) / max(total, 1)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / max(total, 1), 3),
            "avg_overall_score": round(avg_overall, 3),
        },
        "cases": case_results,
        "failure_analysis": _build_failure_analysis(case_results),
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved → {report_path}")
    return report_path


def _build_failure_analysis(case_results: list[dict]) -> dict:
    """Aggregate the top failure modes across all cases."""
    all_weaknesses: list[str] = []
    all_hallucinations: list[str] = []
    all_missed: list[str] = []

    for cr in case_results:
        res = cr["result"]
        all_weaknesses.extend(res["reasoning"].get("weaknesses", []))
        all_hallucinations.extend(res["reasoning"].get("hallucinations", []))
        all_missed.extend(res["recall"].get("missed_key_precedents", []))

    return {
        "top_reasoning_weaknesses": all_weaknesses[:10],
        "hallucinations": all_hallucinations,
        "most_commonly_missed_precedents": all_missed[:10],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> int:
    import sys
    import io

    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.WARNING)

    print("Lexi Legal Agent - Evaluation Suite")
    print(f"Running {len(ALL_CASES)} benchmark case(s)...\n")

    agent, evaluator, doc_repository, engine = await _bootstrap()

    case_results: list[dict] = []
    try:
        for case in ALL_CASES:
            try:
                cr = await _run_case(case, agent, evaluator, doc_repository)
                case_results.append(cr)
            except Exception as exc:  # noqa: BLE001
                _logger.exception("Case %s failed with exception", case.case_id)
                case_results.append(
                    {
                        "case_id": case.case_id,
                        "passed": False,
                        "error": str(exc),
                        "result": {},
                    }
                )
    finally:
        await engine.dispose()
        # Clean up temporary Qdrant directory (gracefully ignore locked files)
        global _TEMP_QDRANT_DIR
        if _TEMP_QDRANT_DIR is not None:
            try:
                _TEMP_QDRANT_DIR.cleanup()
            except Exception:  # noqa: BLE001
                pass  # Ignore cleanup errors from locked files
            _TEMP_QDRANT_DIR = None

    report_path = _write_report(case_results)

    passed = sum(1 for r in case_results if r.get("passed"))
    total = len(case_results)
    print(f"\n{'=' * 60}")
    print(f"FINAL: {passed}/{total} cases passed")
    print(f"{'=' * 60}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
