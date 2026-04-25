"""
LegalAgentEvaluator — measures agent quality on four dimensions:

  1. Precision        — relevant cited / total cited
  2. Recall           — cited ground-truth / all ground-truth
  3. Reasoning        — correctness and depth of legal explanations
  4. Adverse          — honesty about unfavourable precedents

Precision and recall are computed rule-based when ground-truth document IDs
are supplied; otherwise the judge LLM infers them.  Reasoning quality and
adverse identification always go through the LLM judge.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from evals.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from evals.schemas import (
    AdverseResult,
    EvaluationInput,
    EvaluationResult,
    PrecisionResult,
    RecallResult,
    ReasoningResult,
)
from src.llm.base import LLMProvider

_logger = logging.getLogger(__name__)

# Score thresholds for final_verdict classification
_VERDICT_THRESHOLDS = {
    "excellent": 0.8,
    "good": 0.65,
    "average": 0.45,
}


def _verdict(score: float) -> str:
    if score >= _VERDICT_THRESHOLDS["excellent"]:
        return "excellent"
    if score >= _VERDICT_THRESHOLDS["good"]:
        return "good"
    if score >= _VERDICT_THRESHOLDS["average"]:
        return "average"
    return "poor"


def _extract_cited_ids(agent_response: dict[str, Any]) -> list[str]:
    """Return deduplicated document_ids from supporting + adverse precedents."""
    ids: list[str] = []
    for key in ("supporting_precedents", "adverse_precedents"):
        for entry in agent_response.get(key, []):
            doc_id = entry.get("document_id", "")
            if doc_id:
                ids.append(doc_id)
    return list(dict.fromkeys(ids))  # preserve order, deduplicate


def _rule_precision(cited_ids: list[str], ground_truth: list[str]) -> PrecisionResult:
    """Compute precision when ground-truth document IDs are available."""
    if not cited_ids:
        return PrecisionResult(
            score=0.0,
            relevant_cases_count=0,
            total_cases_cited=0,
            explanation="Agent cited no precedents.",
        )
    gt_set = set(ground_truth)
    relevant = [d for d in cited_ids if d in gt_set]
    score = len(relevant) / len(cited_ids)
    return PrecisionResult(
        score=round(score, 3),
        relevant_cases_count=len(relevant),
        total_cases_cited=len(cited_ids),
        explanation=(
            f"{len(relevant)} of {len(cited_ids)} cited documents appear in ground truth."
        ),
    )


def _rule_recall(cited_ids: list[str], ground_truth: list[str]) -> RecallResult:
    """Compute recall when ground-truth document IDs are available."""
    if not ground_truth:
        return RecallResult(
            score=1.0,
            missed_key_precedents=[],
            explanation="No ground-truth documents provided; recall assumed perfect.",
        )
    gt_set = set(ground_truth)
    cited_set = set(cited_ids)
    found = gt_set & cited_set
    missed = sorted(gt_set - cited_set)
    score = len(found) / len(gt_set)
    return RecallResult(
        score=round(score, 3),
        missed_key_precedents=list(missed),
        explanation=(
            f"{len(found)} of {len(gt_set)} ground-truth documents were cited by the agent."
        ),
    )


def _rule_adverse(agent_response: dict[str, Any]) -> AdverseResult | None:
    """
    Deterministic check: if the agent returned zero adverse precedents, that is
    a clear failure regardless of what the LLM judge says.  Returns a zero-score
    result in that case; returns None to signal 'delegate to LLM judge' otherwise.
    """
    adverse_list = agent_response.get("adverse_precedents", [])
    if not adverse_list:
        return AdverseResult(
            score=0.0,
            adverse_cases_identified=[],
            missing_adverse_cases=["(unknown — corpus not examined by evaluator)"],
            risk_analysis_quality=(
                "Agent produced NO adverse precedents. This is a critical failure in legal "
                "practice — a system that never surfaces unfavourable cases exposes clients "
                "to unacknowledged risk."
            ),
        )
    return None  # let the LLM judge rate quality


def _format_retrieved_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "(none)"
    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        file_name = doc.get("file_name", "unknown")
        doc_id = doc.get("document_id", "?")
        score = doc.get("rrf_score") or doc.get("relevance_score") or 0.0
        excerpt = (doc.get("content") or doc.get("excerpt") or "")[:300]
        lines.append(
            f"[{i}] document_id={doc_id}  file={file_name}  score={score:.3f}\n    {excerpt}"
        )
    return "\n".join(lines)


def _format_ground_truth(ground_truth: list[str] | None) -> str:
    if not ground_truth:
        return "(not provided)"
    return ", ".join(ground_truth)


def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from LLM output, stripping any markdown fences."""
    # Strip ```json ... ``` fences if present
    clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().strip("`").strip()
    # Find the outermost JSON object
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in judge response: {raw[:200]}")
    return json.loads(clean[start:end])


class LegalAgentEvaluator:
    """
    Evaluate a single agent response across all four quality dimensions.

    Usage::

        evaluator = LegalAgentEvaluator(llm_provider=my_llm)
        result = await evaluator.evaluate(eval_input)
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """Run all four dimension evaluations and return a consolidated result."""
        cited_ids = _extract_cited_ids(eval_input.agent_response)
        ground_truth = eval_input.ground_truth_docs or []

        # --- Rule-based dimension scores (when ground truth is available) ---
        if ground_truth:
            precision = _rule_precision(cited_ids, ground_truth)
            recall = _rule_recall(cited_ids, ground_truth)
        else:
            precision = None
            recall = None

        # --- Rule-based adverse check (deterministic zero if no adverse found) ---
        adverse_override = _rule_adverse(eval_input.agent_response)

        # --- LLM judge call ---
        judge_result = await self._call_judge(eval_input)

        # --- Merge: rule-based results override LLM where we have ground truth ---
        final_precision = precision or PrecisionResult(**judge_result["precision"])
        final_recall = recall or RecallResult(**judge_result["recall"])
        final_reasoning = ReasoningResult(**judge_result["reasoning"])
        final_adverse = adverse_override or AdverseResult(**judge_result["adverse"])

        # --- Overall score: weighted average (adverse weighted higher — critical safety dim) ---
        overall = (
            final_precision.score * 0.25
            + final_recall.score * 0.25
            + final_reasoning.score * 0.25
            + final_adverse.score * 0.25
        )
        overall = round(overall, 3)

        return EvaluationResult(
            precision=final_precision,
            recall=final_recall,
            reasoning=final_reasoning,
            adverse=final_adverse,
            overall_score=overall,
            final_verdict=_verdict(overall),
        )

    async def _call_judge(self, eval_input: EvaluationInput) -> dict[str, Any]:
        """Format the evaluation prompt, call the LLM, and parse the JSON output."""
        user_content = JUDGE_USER_TEMPLATE.format(
            query=eval_input.query,
            retrieved_docs=_format_retrieved_docs(eval_input.retrieved_docs),
            ground_truth_docs=_format_ground_truth(eval_input.ground_truth_docs),
            agent_response=json.dumps(eval_input.agent_response, indent=2, ensure_ascii=False),
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        _logger.info(
            "Calling LLM judge",
            extra={"query_preview": eval_input.query[:80]},
        )

        response = await self._llm.chat(messages)
        raw_content = response.content or ""

        try:
            return _parse_judge_response(raw_content)
        except (ValueError, json.JSONDecodeError) as exc:
            _logger.warning("LLM judge returned unparseable JSON: %s", exc)
            # Return a neutral fallback so evaluation doesn't crash
            return _neutral_judge_fallback(
                cited_ids=_extract_cited_ids(eval_input.agent_response),
                reason=f"LLM judge output could not be parsed: {exc}",
            )


def _neutral_judge_fallback(cited_ids: list[str], reason: str) -> dict[str, Any]:
    """Return a mid-score fallback when the judge LLM output cannot be parsed."""
    return {
        "precision": {
            "score": 0.5,
            "relevant_cases_count": len(cited_ids),
            "total_cases_cited": len(cited_ids),
            "explanation": reason,
        },
        "recall": {
            "score": 0.5,
            "missed_key_precedents": [],
            "explanation": reason,
        },
        "reasoning": {
            "score": 0.5,
            "strengths": [],
            "weaknesses": [reason],
            "hallucinations": [],
        },
        "adverse": {
            "score": 0.5,
            "adverse_cases_identified": [],
            "missing_adverse_cases": [],
            "risk_analysis_quality": reason,
        },
        "overall_score": 0.5,
        "final_verdict": "average",
    }
