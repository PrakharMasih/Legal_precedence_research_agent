"""
Dimension 1: Precision

Of the precedents the agent identifies as relevant, what percentage are
actually relevant to the case?

Tests cover:
  - Perfect precision (all cited docs in ground truth)
  - Zero precision (no cited docs in ground truth)
  - Partial precision (mixed)
  - Agent cites zero precedents
  - Rule-based path overrides LLM judge when ground truth is provided
"""

from __future__ import annotations

import json

import pytest

from evals.evaluator import LegalAgentEvaluator, _rule_precision
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


# ---------------------------------------------------------------------------
# Unit tests for _rule_precision (no LLM needed)
# ---------------------------------------------------------------------------


def test_rule_precision_perfect():
    result = _rule_precision(["doc-001", "doc-002"], ["doc-001", "doc-002"])
    assert result.score == 1.0
    assert result.relevant_cases_count == 2
    assert result.total_cases_cited == 2


def test_rule_precision_zero():
    result = _rule_precision(["doc-999"], ["doc-001"])
    assert result.score == 0.0
    assert result.relevant_cases_count == 0


def test_rule_precision_partial():
    result = _rule_precision(["doc-001", "doc-999"], ["doc-001"])
    assert result.score == pytest.approx(0.5, abs=0.01)
    assert result.relevant_cases_count == 1
    assert result.total_cases_cited == 2


def test_rule_precision_no_citations():
    result = _rule_precision([], ["doc-001"])
    assert result.score == 0.0
    assert result.total_cases_cited == 0


# ---------------------------------------------------------------------------
# Integration tests via LegalAgentEvaluator (with stub LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_precision_rule_based_overrides_llm():
    """When ground-truth IDs are provided, rule-based precision is used, not LLM."""
    # LLM would say 0.5, but ground truth gives 1.0
    stub_llm = StubLLM(_good_judge_json(precision=0.5))
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    response = _make_research_response(
        supporting=[SUPPORTING_PRECEDENT],
        adverse=[ADVERSE_PRECEDENT],
    )
    eval_input = EvaluationInput(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        agent_response=response,
        ground_truth_docs=["doc-001", "doc-002"],
    )

    result = await evaluator.evaluate(eval_input)

    # Rule-based: both doc-001 and doc-002 are in ground truth → precision = 1.0
    assert result.precision.score == 1.0


@pytest.mark.asyncio
async def test_evaluator_precision_falls_back_to_llm_without_ground_truth():
    """Without ground truth, precision comes from the LLM judge."""
    stub_llm = StubLLM(_good_judge_json(precision=0.75))
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    response = _make_research_response(
        supporting=[SUPPORTING_PRECEDENT],
        adverse=[ADVERSE_PRECEDENT],
    )
    eval_input = EvaluationInput(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        agent_response=response,
        ground_truth_docs=None,
    )

    result = await evaluator.evaluate(eval_input)
    assert result.precision.score == pytest.approx(0.75, abs=0.001)


@pytest.mark.asyncio
async def test_evaluator_precision_irrelevant_citation_lowers_score():
    """An irrelevant document in the citation list reduces rule-based precision."""
    stub_llm = StubLLM(_good_judge_json())
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    irrelevant_precedent = {**SUPPORTING_PRECEDENT, "document_id": "doc-irrelevant"}
    response = _make_research_response(
        supporting=[SUPPORTING_PRECEDENT, irrelevant_precedent],
    )
    eval_input = EvaluationInput(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        agent_response=response,
        ground_truth_docs=["doc-001"],  # only doc-001 is relevant
    )

    result = await evaluator.evaluate(eval_input)
    assert result.precision.score == pytest.approx(0.5, abs=0.01)
    assert result.precision.total_cases_cited == 2
    assert result.precision.relevant_cases_count == 1
