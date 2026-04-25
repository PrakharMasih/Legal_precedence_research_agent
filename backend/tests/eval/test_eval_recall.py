"""
Dimension 2: Recall

Of the precedents that should have been found, what percentage did the agent
actually find?

Tests cover:
  - Perfect recall (all ground-truth docs cited)
  - Zero recall (no ground-truth docs cited)
  - Partial recall
  - No ground-truth provided (defaults to LLM judge, assumed perfect in rule-based)
  - Missed precedents are correctly listed
"""

from __future__ import annotations

import pytest

from evals.evaluator import LegalAgentEvaluator, _rule_recall
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
# Unit tests for _rule_recall
# ---------------------------------------------------------------------------


def test_rule_recall_perfect():
    result = _rule_recall(["doc-001", "doc-002"], ["doc-001", "doc-002"])
    assert result.score == 1.0
    assert result.missed_key_precedents == []


def test_rule_recall_zero():
    result = _rule_recall([], ["doc-001", "doc-002"])
    assert result.score == 0.0
    assert set(result.missed_key_precedents) == {"doc-001", "doc-002"}


def test_rule_recall_partial():
    result = _rule_recall(["doc-001"], ["doc-001", "doc-002"])
    assert result.score == pytest.approx(0.5, abs=0.01)
    assert "doc-002" in result.missed_key_precedents
    assert "doc-001" not in result.missed_key_precedents


def test_rule_recall_empty_ground_truth_returns_perfect():
    """When no ground truth is provided, recall defaults to 1.0 (no evidence of gaps)."""
    result = _rule_recall(["doc-001"], [])
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Integration tests via LegalAgentEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_recall_rule_based_with_ground_truth():
    stub_llm = StubLLM(_good_judge_json(recall=0.3))  # LLM would say 0.3 — should be ignored
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    # Agent cited only doc-001, but ground truth also includes doc-002 and doc-003
    response = _make_research_response(supporting=[SUPPORTING_PRECEDENT])
    eval_input = EvaluationInput(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        agent_response=response,
        ground_truth_docs=["doc-001", "doc-002", "doc-003"],
    )

    result = await evaluator.evaluate(eval_input)

    # Rule-based: 1 of 3 found → recall ≈ 0.333
    assert result.recall.score == pytest.approx(1 / 3, abs=0.01)
    assert "doc-002" in result.recall.missed_key_precedents
    assert "doc-003" in result.recall.missed_key_precedents


@pytest.mark.asyncio
async def test_evaluator_recall_llm_judge_when_no_ground_truth():
    stub_llm = StubLLM(_good_judge_json(recall=0.6))
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
    # No ground truth → LLM judge value used
    assert result.recall.score == pytest.approx(0.6, abs=0.001)


@pytest.mark.asyncio
async def test_evaluator_recall_all_ground_truth_missed():
    stub_llm = StubLLM(_good_judge_json())
    evaluator = LegalAgentEvaluator(llm_provider=stub_llm)

    # Agent cited nothing relevant to ground truth
    irrelevant = {**SUPPORTING_PRECEDENT, "document_id": "doc-irrelevant"}
    response = _make_research_response(supporting=[irrelevant])
    eval_input = EvaluationInput(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        agent_response=response,
        ground_truth_docs=["doc-001", "doc-002"],
    )

    result = await evaluator.evaluate(eval_input)
    assert result.recall.score == 0.0
    assert len(result.recall.missed_key_precedents) == 2
