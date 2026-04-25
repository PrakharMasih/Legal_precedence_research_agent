from __future__ import annotations

from pydantic import BaseModel, Field


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
