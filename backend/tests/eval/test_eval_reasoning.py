"""
Dimension 3: Reasoning Quality

Does the agent correctly explain why each precedent applies or does not apply?

Tests cover:
  - High-quality reasoning: correct principles, factual alignment, no hallucinations
  - Shallow reasoning: vague explanations flagged by judge
  - Hallucinated case names: judge detects invented citations
  - Score reflects strengths and weaknesses arrays from LLM judge
  - Unparseable LLM output triggers neutral fallback
"""

from __future__ import annotations

import json

import pytest

from evals.evaluator import LegalAgentEvaluator
from evals.schemas import EvaluationInput
from src.llm.base import LLMResponse
from tests.eval.conftest import (
    ADVERSE_PRECEDENT,
    QUERY,
    RETRIEVED_DOCS,
    SUPPORTING_PRECEDENT,
    StubLLM,
    _good_judge_json,
    _make_research_response,
)


def _reasoning_judge_json(
    score: float,
    strengths: list[str],
    weaknesses: list[str],
    hallucinations: list[str],
) -> str:
    overall = round((score + 0.7 + 0.7 + 0.7) / 4, 3)
    return json.dumps(
        {
            "precision": {
                "score": 0.7,
                "relevant_cases_count": 1,
                "total_cases_cited": 1,
                "explanation": "ok",
            },
            "recall": {
                "score": 0.7,
                "missed_key_precedents": [],
                "explanation": "ok",
            },
            "reasoning": {
                "score": score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "hallucinations": hallucinations,
            },
            "adverse": {
                "score": 0.7,
                "adverse_cases_identified": [],
                "missing_adverse_cases": [],
                "risk_analysis_quality": "ok",
            },
            "overall_score": overall,
            "final_verdict": "good",
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_good_reasoning_produces_high_score():
    stub_llm = StubLLM(
        _reasoning_judge_json(
            score=0.9,
            strengths=["Legal principle precisely stated", "Factual alignment explicit"],
            weaknesses=[],
            hallucinations=[],
        )
    )
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)
    response = _make_research_response(
        supporting=[SUPPORTING_PRECEDENT], adverse=[ADVERSE_PRECEDENT]
    )
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    assert result.reasoning.score == pytest.approx(0.9, abs=0.001)
    assert result.reasoning.hallucinations == []
    assert len(result.reasoning.strengths) >= 1


@pytest.mark.asyncio
async def test_shallow_reasoning_produces_low_score():
    stub_llm = StubLLM(
        _reasoning_judge_json(
            score=0.3,
            strengths=[],
            weaknesses=["Legal principle vague", "No factual alignment provided"],
            hallucinations=[],
        )
    )
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)
    vague_supporting = {
        **SUPPORTING_PRECEDENT,
        "legal_principle": "Case is relevant.",  # vague
        "factual_alignment": "",
    }
    response = _make_research_response(supporting=[vague_supporting])
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    assert result.reasoning.score == pytest.approx(0.3, abs=0.001)
    assert len(result.reasoning.weaknesses) >= 1


@pytest.mark.asyncio
async def test_hallucinated_case_name_is_flagged():
    stub_llm = StubLLM(
        _reasoning_judge_json(
            score=0.2,
            strengths=[],
            weaknesses=["Invented case name not in corpus"],
            hallucinations=["Sharma v. India (2021) — not found in retrieved documents"],
        )
    )
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)
    hallucinated = {
        **SUPPORTING_PRECEDENT,
        "case_name": "Sharma v. India (2021)",  # not in corpus
        "document_id": "doc-hallucinated",
    }
    response = _make_research_response(supporting=[hallucinated])
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    assert len(result.reasoning.hallucinations) >= 1
    assert result.reasoning.score < 0.5


@pytest.mark.asyncio
async def test_unparseable_llm_response_triggers_fallback():
    class BrokenLLM:
        async def chat(self, messages, tools=None):
            return LLMResponse(content="NOT JSON AT ALL %%%", tool_calls=[], raw={})

    evaluator = LegalAgentEvaluator(llm_provider=BrokenLLM())
    # Include an adverse precedent so the rule-based adverse override does NOT
    # zero-out the adverse dimension — this isolates the reasoning fallback path.
    response = _make_research_response(
        supporting=[SUPPORTING_PRECEDENT], adverse=[ADVERSE_PRECEDENT]
    )
    result = await evaluator.evaluate(
        EvaluationInput(query=QUERY, retrieved_docs=RETRIEVED_DOCS, agent_response=response)
    )

    # Neutral fallback: all LLM-judged dimensions default to 0.5 → overall 0.5 → "average"
    assert result.reasoning.score == pytest.approx(0.5, abs=0.001)
    assert result.overall_score == pytest.approx(0.5, abs=0.01)
    assert result.final_verdict == "average"
