"""
Dimension 4: Adverse Precedent Identification

Did the agent surface judgments that work against the client's position?

Tests cover:
  - No adverse precedents at all → deterministic score=0 (critical failure)
  - Adverse precedents present → LLM judge rates quality
  - Risks clearly articulated → high score
  - Risks downplayed or missing distinguishing arguments → lower score
  - Overall score is penalised when adverse dimension fails
"""

from __future__ import annotations

import json

import pytest

from evals.evaluator import LegalAgentEvaluator, _rule_adverse
from evals.schemas import EvaluationInput
from tests.eval.conftest import (
    ADVERSE_PRECEDENT,
    QUERY,
    RETRIEVED_DOCS,
    SUPPORTING_PRECEDENT,
    StubLLM,
    _good_judge_json,
    _make_research_response,
)


def _adverse_judge_json(
    score: float, identified: list[str], missing: list[str], quality: str
) -> str:
    overall = round((0.7 + 0.7 + 0.7 + score) / 4, 3)
    return json.dumps(
        {
            "precision": {
                "score": 0.7,
                "relevant_cases_count": 1,
                "total_cases_cited": 1,
                "explanation": "ok",
            },
            "recall": {"score": 0.7, "missed_key_precedents": [], "explanation": "ok"},
            "reasoning": {
                "score": 0.7,
                "strengths": [],
                "weaknesses": [],
                "hallucinations": [],
            },
            "adverse": {
                "score": score,
                "adverse_cases_identified": identified,
                "missing_adverse_cases": missing,
                "risk_analysis_quality": quality,
            },
            "overall_score": overall,
            "final_verdict": "good" if overall >= 0.65 else "average",
        }
    )


# ---------------------------------------------------------------------------
# Unit test for _rule_adverse
# ---------------------------------------------------------------------------


def test_rule_adverse_returns_zero_when_no_adverse_cases():
    response = _make_research_response(supporting=[SUPPORTING_PRECEDENT], adverse=[])
    result = _rule_adverse(response)
    assert result is not None
    assert result.score == 0.0
    assert "NO adverse precedents" in result.risk_analysis_quality


def test_rule_adverse_returns_none_when_adverse_cases_present():
    """None signals 'delegate quality rating to LLM judge'."""
    response = _make_research_response(adverse=[ADVERSE_PRECEDENT])
    result = _rule_adverse(response)
    assert result is None


# ---------------------------------------------------------------------------
# Integration tests via LegalAgentEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_adverse_cases_forces_score_to_zero_regardless_of_llm():
    """Even if the LLM judge would give a passing score, zero adverse = 0.0."""
    # LLM would give 0.8 — but rule-based override must set 0.0
    stub_llm = StubLLM(_good_judge_json(adverse=0.8))
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    response = _make_research_response(supporting=[SUPPORTING_PRECEDENT], adverse=[])
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    assert result.adverse.score == 0.0
    assert "NO adverse precedents" in result.adverse.risk_analysis_quality


@pytest.mark.asyncio
async def test_adverse_cases_present_score_comes_from_llm():
    """When adverse cases exist, quality score comes from the LLM judge."""
    stub_llm = StubLLM(
        _adverse_judge_json(
            score=0.85,
            identified=["Oriental Insurance Co v Nanjappan"],
            missing=[],
            quality="Risk clearly described; distinguishing argument provided.",
        )
    )
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)
    response = _make_research_response(
        supporting=[SUPPORTING_PRECEDENT], adverse=[ADVERSE_PRECEDENT]
    )
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    assert result.adverse.score == pytest.approx(0.85, abs=0.001)
    assert "Oriental Insurance Co v Nanjappan" in result.adverse.adverse_cases_identified


@pytest.mark.asyncio
async def test_downplayed_risks_produce_lower_adverse_score():
    stub_llm = StubLLM(
        _adverse_judge_json(
            score=0.3,
            identified=["Oriental Insurance Co v Nanjappan"],
            missing=["Contributory negligence cases not mentioned"],
            quality="Risk mentioned but severity not quantified; no distinguishing argument.",
        )
    )
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)
    weak_adverse = {
        **ADVERSE_PRECEDENT,
        "risk_description": "This case might be a problem.",  # vague
        "distinguishing_argument": "",  # missing
    }
    response = _make_research_response(adverse=[weak_adverse])
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    assert result.adverse.score == pytest.approx(0.3, abs=0.001)
    assert len(result.adverse.missing_adverse_cases) >= 1


@pytest.mark.asyncio
async def test_overall_score_penalised_when_adverse_fails():
    """Zero adverse score drags overall below passing threshold."""
    # Precision, recall, reasoning all perfect — but adverse = 0
    stub_llm = StubLLM(_good_judge_json(precision=1.0, recall=1.0, reasoning=1.0, adverse=1.0))
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    response = _make_research_response(supporting=[SUPPORTING_PRECEDENT], adverse=[])
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    # Adverse is overridden to 0.0 → overall = (1+1+1+0)/4 = 0.75 even with perfect other scores
    assert result.adverse.score == 0.0
    assert result.overall_score == pytest.approx(0.75, abs=0.01)
