"""Pydantic models for evaluation inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dimension result models (mirror the JSON schema in the eval prompt)
# ---------------------------------------------------------------------------


class PrecisionResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    relevant_cases_count: int
    total_cases_cited: int
    explanation: str


class RecallResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    missed_key_precedents: list[str]
    explanation: str


class ReasoningResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    strengths: list[str]
    weaknesses: list[str]
    hallucinations: list[str]


class AdverseResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    adverse_cases_identified: list[str]
    missing_adverse_cases: list[str]
    risk_analysis_quality: str


class EvaluationResult(BaseModel):
    precision: PrecisionResult
    recall: RecallResult
    reasoning: ReasoningResult
    adverse: AdverseResult
    overall_score: float = Field(ge=0.0, le=1.0)
    final_verdict: Literal["poor", "average", "good", "excellent"]


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class EvaluationInput(BaseModel):
    """All data required to run a single evaluation."""

    query: str
    # Serialised list of RankedChunk-like dicts (file_name, content, document_id, …)
    retrieved_docs: list[dict[str, Any]]
    # Serialised PrecedentAnalysis or GeneralQueryResponse dict from the agent
    agent_response: dict[str, Any]
    # Optional ground-truth document IDs; enables rule-based precision/recall
    ground_truth_docs: list[str] | None = None


# ---------------------------------------------------------------------------
# Benchmark case model
# ---------------------------------------------------------------------------


class BenchmarkCase(BaseModel):
    """One labelled test case used by the evaluation runner."""

    case_id: str
    query: str
    # IDs of documents that *should* appear in a correct response
    ground_truth_doc_ids: list[str] = Field(default_factory=list)
    # Legal themes that *must* be addressed (used for recall inference)
    expected_themes: list[str] = Field(default_factory=list)
    # Themes that indicate adverse precedents exist in the corpus for this case
    expected_adverse_themes: list[str] = Field(default_factory=list)
    # Minimum acceptable scores (used in automated pass/fail assertions)
    min_precision: float = 0.5
    min_recall: float = 0.5
    min_reasoning: float = 0.5
    min_adverse: float = 0.5
