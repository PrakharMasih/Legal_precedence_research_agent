"""Shared fixtures for eval tests."""

from __future__ import annotations

from typing import Any

import pytest

from evals.schemas import EvaluationInput
from src.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Minimal agent responses for testing
# ---------------------------------------------------------------------------


def _make_research_response(
    supporting: list[dict] | None = None,
    adverse: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "supporting_precedents": supporting or [],
        "adverse_precedents": adverse or [],
        "strategy_recommendation": {
            "priority_arguments": [],
            "compensation_range": "₹20–40 lakh",
            "risks": [],
        },
    }


SUPPORTING_PRECEDENT = {
    "document_id": "doc-001",
    "file_name": "swaran_singh.pdf",
    "case_name": "National Insurance Co v Swaran Singh",
    "excerpt": "Insurer remained liable to third-party claimant despite licence breach.",
    "legal_principle": "Insurers cannot escape third-party liability on policy-condition grounds.",
    "factual_alignment": "Matches: unlicensed driver, insurer denial, third-party victim.",
}

ADVERSE_PRECEDENT = {
    "document_id": "doc-002",
    "file_name": "oriental_insurance.pdf",
    "case_name": "Oriental Insurance Co v Nanjappan",
    "excerpt": "Policy exclusion for driving without licence is valid against owner.",
    "risk_description": "Insurer may invoke exclusion clause against the vehicle owner.",
    "distinguishing_argument": "Swaran Singh establishes pay-and-recover — insurer still pays "
    "third party and recovers from owner.",
}

QUERY = (
    "Mrs. Lakshmi Devi was injured by an unlicensed truck driver. "
    "The insurer denied liability. What precedents support her claim?"
)

RETRIEVED_DOCS = [
    {
        "document_id": "doc-001",
        "file_name": "swaran_singh.pdf",
        "content": "Insurer remained liable to third-party claimant despite licence breach.",
        "rrf_score": 0.85,
    },
    {
        "document_id": "doc-002",
        "file_name": "oriental_insurance.pdf",
        "content": "Policy exclusion for driving without licence is valid against owner.",
        "rrf_score": 0.71,
    },
]


# ---------------------------------------------------------------------------
# Stub LLM that returns a configurable response
# ---------------------------------------------------------------------------


class StubLLM:
    """Returns a fixed JSON string as the LLM judge's response."""

    def __init__(self, judge_json: str) -> None:
        self._json = judge_json

    async def chat(self, messages, tools=None) -> LLMResponse:
        return LLMResponse(content=self._json, tool_calls=[], raw={})


def _good_judge_json(
    precision: float = 0.9,
    recall: float = 0.85,
    reasoning: float = 0.8,
    adverse: float = 0.75,
) -> str:
    import json

    overall = round((precision + recall + reasoning + adverse) / 4, 3)
    verdict = "excellent" if overall >= 0.8 else "good" if overall >= 0.65 else "average"
    return json.dumps(
        {
            "precision": {
                "score": precision,
                "relevant_cases_count": 2,
                "total_cases_cited": 2,
                "explanation": "Both cited cases directly address the unlicensed-driver issue.",
            },
            "recall": {
                "score": recall,
                "missed_key_precedents": [],
                "explanation": "All key angles covered.",
            },
            "reasoning": {
                "score": reasoning,
                "strengths": ["Correct legal principle stated", "Factual alignment precise"],
                "weaknesses": [],
                "hallucinations": [],
            },
            "adverse": {
                "score": adverse,
                "adverse_cases_identified": ["Oriental Insurance Co v Nanjappan"],
                "missing_adverse_cases": [],
                "risk_analysis_quality": "Risk clearly explained with distinguishing argument.",
            },
            "overall_score": overall,
            "final_verdict": verdict,
        }
    )
