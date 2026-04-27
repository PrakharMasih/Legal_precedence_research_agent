from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.output_schemas import (
    AdversePrecedent,
    GeneralQueryResponse,
    PrecedentAnalysis,
    StrategyRecommendation,
    SupportingDocument,
    SupportingPrecedent,
)
from src.models.query import RankedChunk, SearchMode
from src.retrieval.retriever import Retriever
from src.storage.repositories import ChunkRepository, DocumentRepository


@dataclass(slots=True)
class ResearchToolbox:
    retriever: Retriever
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "description": (
                        "Search the indexed corpus for relevant judgment chunks. "
                        "Run multiple calls with different targeted keywords to cover "
                        "distinct legal angles (core issue, doctrine, compensation method, etc.)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Specific legal keywords or phrase to search for.",
                            },
                            "n_results": {
                                "type": "integer",
                                "default": 10,
                                "description": "Number of results to return (5–20).",
                            },
                            "search_mode": {
                                "type": "string",
                                "enum": ["dense", "sparse", "hybrid"],
                                "default": "hybrid",
                                "description": (
                                    "Use 'hybrid' for most queries; "
                                    "'sparse' for exact legal terms or citation numbers."
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_summary",
                    "description": (
                        "Retrieve full metadata and a first excerpt for a specific document "
                        "by its ID. Use this to get richer context for a promising document "
                        "found via search_corpus."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "The document_id from a search_corpus result.",
                            },
                        },
                        "required": ["document_id"],
                    },
                },
            },
        ]

    async def search_corpus(
        self,
        query: str,
        n_results: int = 10,
        search_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        results = await self.retriever.retrieve(
            query,
            n=n_results,
            search_mode=SearchMode(search_mode),
        )
        return [self._serialize_ranked_chunk(result) for result in results]

    async def get_document_summary(self, document_id: str) -> dict[str, Any]:
        document = await self.document_repository.get_by_id(document_id)
        if document is None:
            return {}
        chunks = await self.chunk_repository.get_by_document_id(document_id)
        excerpt = chunks[0].content if chunks else ""
        return {
            "document_id": document.id,
            "file_name": document.file_name,
            "case_name": document.case_name,
            "excerpt": excerpt,
        }

    async def build_precedent_entry(
        self,
        document_id: str,
        role: str,
        reasoning: str,
    ) -> dict[str, Any]:
        summary = await self.get_document_summary(document_id)
        if role == "supporting":
            return SupportingPrecedent(
                document_id=document_id,
                file_name=summary.get("file_name", ""),
                case_name=summary.get("case_name"),
                excerpt=summary.get("excerpt"),
                legal_principle=reasoning,
                factual_alignment=reasoning,
            ).model_dump()

        return AdversePrecedent(
            document_id=document_id,
            file_name=summary.get("file_name", ""),
            case_name=summary.get("case_name"),
            excerpt=summary.get("excerpt"),
            risk_description=reasoning,
            distinguishing_argument=(
                "Distinguish on the specific statutory and factual context of the claim."
            ),
        ).model_dump()

    def finalize_research_response(
        self,
        supporting: list[dict[str, Any]],
        adverse: list[dict[str, Any]],
        strategy: dict[str, Any],
    ) -> dict[str, Any]:
        return PrecedentAnalysis(
            supporting_precedents=supporting,
            adverse_precedents=adverse,
            strategy_recommendation=StrategyRecommendation.model_validate(strategy),
        ).model_dump()

    def finalize_general_response(
        self,
        answer: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = [SupportingDocument.model_validate(document) for document in documents]
        normalized.sort(key=lambda item: item.relevance_score, reverse=True)
        return GeneralQueryResponse(answer=answer, supporting_documents=normalized).model_dump()

    def _serialize_ranked_chunk(self, result: RankedChunk) -> dict[str, Any]:
        return {
            "document_id": result.document_id,
            "file_name": result.file_name,
            "case_name": result.case_name,
            "section": result.section,
            # Prefer parent-context window for richer LLM reasoning; fall back to child content
            "excerpt": result.parent_content or result.content,
            "relevance_score": max(0.0, min(1.0, result.rrf_score * 100)),
        }
