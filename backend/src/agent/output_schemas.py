from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Planner schemas
# ---------------------------------------------------------------------------


class PlannerOutput(BaseModel):
    """Structured execution plan produced by the Planner node."""

    query_type: str  # "precedent_research" | "general_query" | "conversational"
    requires_retrieval: bool
    depth: str  # "shallow" | "medium" | "deep"
    sub_queries: list[str] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    strategy: str  # "multi_step_research" | "direct_answer" | "conversational"


# ---------------------------------------------------------------------------
# IRAC Reasoning schemas
# ---------------------------------------------------------------------------


class ContradictionNote(BaseModel):
    """Two precedents that conflict on the same legal point."""

    doc_id_a: str
    doc_id_b: str
    description: str


class IRACReasoning(BaseModel):
    """Structured IRAC analysis produced by the Reasoner node."""

    issue: str
    applicable_rules: list[str] = Field(default_factory=list)
    application: str
    preliminary_conclusion: str
    precedent_strengths: dict[str, float] = Field(default_factory=dict)
    contradictions: list[ContradictionNote] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reflection schemas
# ---------------------------------------------------------------------------


class ReflectionResult(BaseModel):
    """Self-assessment produced by the Reflector node."""

    confidence: float  # 0.0 – 1.0
    reasoning_quality: str  # "sufficient" | "needs_improvement" | "insufficient"
    missing_aspects: list[str] = Field(default_factory=list)
    needs_more_retrieval: bool
    refinement_queries: list[str] = Field(default_factory=list)
    contradictions_addressed: bool


# ---------------------------------------------------------------------------
# Existing precedent / synthesis schemas
# ---------------------------------------------------------------------------


class SupportingPrecedent(BaseModel):
    document_id: str
    file_name: str
    case_name: str | None = None
    excerpt: str | None = None
    legal_principle: str
    factual_alignment: str


class AdversePrecedent(BaseModel):
    document_id: str
    file_name: str
    case_name: str | None = None
    excerpt: str | None = None
    risk_description: str
    distinguishing_argument: str


class StrategyRecommendation(BaseModel):
    priority_arguments: list[str] = Field(default_factory=list)
    compensation_range: str
    risks: list[str] = Field(default_factory=list)


class PrecedentAnalysis(BaseModel):
    supporting_precedents: list[SupportingPrecedent] = Field(default_factory=list)
    adverse_precedents: list[AdversePrecedent] = Field(default_factory=list)
    strategy_recommendation: StrategyRecommendation


class SupportingDocument(BaseModel):
    document_id: str
    file_name: str
    case_name: str | None = None
    excerpt: str | None = None
    relevance_score: float


class GeneralQueryResponse(BaseModel):
    answer: str
    supporting_documents: list[SupportingDocument] = Field(default_factory=list)
