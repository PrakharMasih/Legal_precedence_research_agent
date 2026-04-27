"""
Graph nodes for the autonomous legal reasoning agent.

Pipeline:
    PlannerNode → RetrievalNode → ReasonerNode → ReflectorNode → SynthesisNode

Each node accepts the shared AgentState, modifies it in-place, and emits events.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from src.agent.graph.state import AgentState
from src.agent.output_schemas import (
    AdversePrecedent,
    IRACReasoning,
    PlannerOutput,
    PrecedentAnalysis,
    ReflectionResult,
    StrategyRecommendation,
    SupportingPrecedent,
)
from src.agent.prompts import (
    IRAC_REASONING_PROMPT,
    PLANNER_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CHAT_SYSTEM_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
)
from src.agent.tools import ResearchToolbox
from src.core.exceptions import LLMUnavailableError
from src.llm.base import LLMProvider

_INTER_REQUEST_DELAY = 1.0  # seconds between consecutive LLM calls (Groq rate-limit guard)
_MAX_CONTEXT_CHUNKS = 15  # unique documents forwarded to synthesis


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_json_block(raw: str) -> dict[str, Any] | None:
    """Extract and parse the first JSON object found in an LLM response."""
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Render a numbered context block for LLM prompts."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        case_name = chunk.get("case_name") or "Unknown"
        parts.append(
            f"[{i}] CASE: {case_name}\n"
            f"Document: {chunk.get('file_name', 'unknown')} (ID: {chunk.get('document_id', '')})\n"
            f"Relevance: {chunk.get('relevance_score', 0):.3f} | "
            f"Section: {chunk.get('section', 'N/A')}\n"
            f"Excerpt:\n{chunk.get('excerpt', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _format_irac_summary(irac: IRACReasoning | None) -> str:
    """Compact IRAC text block injected into synthesis / general-answer prompts."""
    if not irac:
        return ""
    lines = [
        f"Issue: {irac.issue}",
        "Applicable Rules:",
        *[f"  - {rule}" for rule in irac.applicable_rules],
        f"Application: {irac.application}",
        f"Preliminary Conclusion: {irac.preliminary_conclusion}",
    ]
    if irac.contradictions:
        lines.append("Contradictions Detected:")
        for c in irac.contradictions:
            lines.append(f"  - {c.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PlannerNode
# ---------------------------------------------------------------------------


class PlannerNode:
    """
    Phase 1 — Query decomposition.

    Calls the LLM once with PLANNER_PROMPT to produce a PlannerOutput that
    specifies query_type, depth, and a list of diverse sub_queries. If the
    LLM fails or returns invalid JSON, a keyword-based fallback is used.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def execute(self, state: AgentState) -> None:
        await state.emit(
            {
                "type": "thinking",
                "phase": "planning",
                "message": "Analyzing query and building execution plan…",
            }
        )
        try:
            # Include recent history so the planner understands follow-up references
            # like "these judgments" or "those cases" from prior turns.
            history_msgs = [
                {"role": m["role"], "content": str(m.get("content", ""))[:600]}
                for m in state.history[-4:]
                if m.get("role") in ("user", "assistant")
            ]
            resp = await self._llm.chat(
                [
                    {"role": "system", "content": PLANNER_PROMPT},
                    *history_msgs,
                    {"role": "user", "content": state.query_text},
                ]
            )
            data = _parse_json_block(resp.content)
            if data:
                plan = PlannerOutput.model_validate(data)
                state.plan = plan
                sub_q_preview = " | ".join(f'"{q[:60]}"' for q in plan.sub_queries)
                await state.emit(
                    {
                        "type": "thinking",
                        "phase": "planning",
                        "message": (
                            f"Plan: {plan.strategy} | depth={plan.depth} | "
                            f"{len(plan.sub_queries)} sub-queries: {sub_q_preview}"
                        ),
                        "plan": plan.model_dump(),
                    }
                )
                return
        except LLMUnavailableError:
            raise
        except Exception:  # noqa: BLE001
            pass

        fallback = self._fallback_plan(state.query_text)
        state.plan = fallback
        await state.emit(
            {
                "type": "thinking",
                "phase": "planning",
                "message": f"Fallback plan: {fallback.strategy}",
                "plan": fallback.model_dump(),
            }
        )

    @staticmethod
    def _fallback_plan(query_text: str) -> PlannerOutput:
        lowered = query_text.lower()

        # Follow-up references to already-retrieved material → always general_query
        is_follow_up = any(
            phrase in lowered
            for phrase in (
                "which of these",
                "these judgments",
                "those cases",
                "those judgments",
                "the above cases",
                "the retrieved",
                "any of these",
                "from the results",
            )
        )

        # Only strongly adversarial / strategic keywords signal precedent_research.
        # Generic words like "judgment" or "court" also appear in informational questions
        # and must NOT trigger research mode.
        is_research = not is_follow_up and any(
            phrase in lowered
            for phrase in (
                "strategy",
                "precedent",
                "adverse precedent",
                "support my case",
                "support our case",
                "argue that",
                "case law",
                "claim compensation",
                "litigation strategy",
                "will i win",
                "how strong is",
                "help me argue",
                "find cases for",
                "find precedents",
            )
        )
        return PlannerOutput(
            query_type="precedent_research" if is_research else "general_query",
            requires_retrieval=True,
            depth="medium" if is_research else "shallow",
            sub_queries=[query_text],
            legal_issues=[query_text],
            strategy="multi_step_research" if is_research else "direct_answer",
        )


# ---------------------------------------------------------------------------
# RetrievalNode
# ---------------------------------------------------------------------------


class RetrievalNode:
    """
    Phase 2 — Deterministic multi-query corpus search.

    Runs one search_corpus call per sub_query from the plan (no LLM involved).
    Results are accumulated across reflection-loop iterations and deduplicated
    by document_id, keeping the highest-scoring chunk per document.
    """

    def __init__(self, toolbox: ResearchToolbox) -> None:
        self._toolbox = toolbox

    async def execute(self, state: AgentState) -> None:
        sub_queries = (
            state.plan.sub_queries if state.plan and state.plan.sub_queries else [state.query_text]
        )
        n_results = 12 if state.retrieval_iteration == 0 else 8

        await state.emit(
            {
                "type": "thinking",
                "phase": "retrieval",
                "message": (
                    f"[Iteration {state.retrieval_iteration + 1}] "
                    f"Running {len(sub_queries)} targeted corpus searches…"
                ),
                "sub_queries": sub_queries,
            }
        )

        new_retrieved: list[dict[str, Any]] = []
        for i, query in enumerate(sub_queries):
            if i > 0:
                await asyncio.sleep(0.3)  # light delay between non-LLM corpus calls
            results = await self._toolbox.search_corpus(
                query=query,
                n_results=n_results,
                search_mode="hybrid",
            )
            new_retrieved.extend(results)
            await state.emit(
                {
                    "type": "tool_result",
                    "tool": "search_corpus",
                    "query": query,
                    "total_returned": len(results),
                    "top_results": [
                        {
                            "file_name": r.get("file_name", ""),
                            "case_name": r.get("case_name", ""),
                            "relevance_score": round(r.get("relevance_score", 0), 3),
                            "excerpt_preview": (r.get("excerpt") or "")[:150],
                        }
                        for r in results[:3]
                    ],
                }
            )

        # Accumulate across iterations then deduplicate
        state.all_retrieved.extend(new_retrieved)

        seen: dict[str, dict[str, Any]] = {}
        for chunk in state.all_retrieved:
            doc_id = chunk.get("document_id", "")
            if not doc_id:
                continue
            if doc_id not in seen or chunk.get("relevance_score", 0) > seen[doc_id].get(
                "relevance_score", 0
            ):
                seen[doc_id] = chunk

        state.deduped_context = sorted(
            seen.values(),
            key=lambda c: c.get("relevance_score", 0),
            reverse=True,
        )[:_MAX_CONTEXT_CHUNKS]

        await state.emit(
            {
                "type": "thinking",
                "phase": "retrieval",
                "message": (
                    f"Retrieved {len(new_retrieved)} chunks → "
                    f"{len(state.deduped_context)} unique documents after deduplication"
                ),
                "total_retrieved": len(state.all_retrieved),
                "unique_documents": len(state.deduped_context),
            }
        )


# ---------------------------------------------------------------------------
# ReasonerNode  (IRAC)
# ---------------------------------------------------------------------------


class ReasonerNode:
    """
    Phase 3 — Structured IRAC legal reasoning.

    Sends the retrieved context to the LLM with IRAC_REASONING_PROMPT,
    which returns:
      • The precise legal issue
      • Rules extracted from each precedent
      • Application of those rules to the query
      • A preliminary conclusion
      • Per-document precedent strength scores (0–1)
      • Contradiction notes between conflicting precedents
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def execute(self, state: AgentState) -> None:
        await state.emit(
            {"type": "reasoning", "message": "Applying IRAC reasoning to retrieved judgments…"}
        )

        context_block = _format_context(state.deduped_context)
        legal_issues = state.plan.legal_issues if state.plan else [state.query_text]

        user_content = (
            f"USER QUERY:\n{state.query_text}\n\n"
            "IDENTIFIED LEGAL ISSUES:\n"
            + "\n".join(f"- {issue}" for issue in legal_issues)
            + f"\n\nRETRIEVED JUDGMENTS:\n{context_block}\n\n"
            "Apply IRAC methodology and produce the JSON analysis."
        )

        try:
            await asyncio.sleep(_INTER_REQUEST_DELAY)
            resp = await self._llm.chat(
                [
                    {"role": "system", "content": IRAC_REASONING_PROMPT},
                    {"role": "user", "content": user_content},
                ]
            )
            data = _parse_json_block(resp.content)
            if data:
                irac = IRACReasoning.model_validate(data)
                state.irac = irac
                issue_preview = irac.issue[:80] + "\u2026" if len(irac.issue) > 80 else irac.issue
                contradiction_note = (
                    f" | \u26a0 {len(irac.contradictions)} conflict(s) detected"
                    if irac.contradictions
                    else ""
                )
                await state.emit(
                    {
                        "type": "reasoning",
                        "message": (
                            f'IRAC: Issue \u2014 "{issue_preview}" | '
                            f"{len(irac.applicable_rules)} rules extracted" + contradiction_note
                        ),
                        "issue": irac.issue,
                        "rules_count": len(irac.applicable_rules),
                        "contradictions": [c.model_dump() for c in irac.contradictions],
                        "precedent_strengths": irac.precedent_strengths,
                    }
                )
                return
        except LLMUnavailableError:
            raise
        except Exception:  # noqa: BLE001
            pass

        # Fallback: derive strengths directly from relevance scores
        state.irac = self._fallback_irac(state.query_text, state.deduped_context)
        await state.emit({"type": "reasoning", "message": "Using fallback IRAC (LLM call failed)."})

    @staticmethod
    def _fallback_irac(query_text: str, context: list[dict[str, Any]]) -> IRACReasoning:
        strengths = {
            c["document_id"]: round(min(1.0, c.get("relevance_score", 0.5)), 2)
            for c in context
            if c.get("document_id")
        }
        return IRACReasoning(
            issue=f"Legal question arising from: {query_text}",
            applicable_rules=["See retrieved judgment excerpts for applicable legal principles."],
            application=("Rules from retrieved judgments should be applied to the query facts."),
            preliminary_conclusion=("See retrieved precedents for guidance on the likely outcome."),
            precedent_strengths=strengths,
            contradictions=[],
        )


# ---------------------------------------------------------------------------
# ReflectorNode
# ---------------------------------------------------------------------------


class ReflectorNode:
    """
    Phase 4 — Self-assessment and loop control.

    Evaluates the IRAC analysis and emits a ReflectionResult with:
      • confidence  (0–1)
      • reasoning_quality
      • missing_aspects   — what's not covered
      • needs_more_retrieval  — whether to loop back
      • refinement_queries    — targeted queries to fill gaps

    If confidence < 0.6 and the loop budget allows, the workflow will
    update sub_queries with refinement_queries and run another retrieval pass.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def execute(self, state: AgentState) -> None:
        await state.emit(
            {
                "type": "thinking",
                "phase": "reflection",
                "message": (
                    "Self-evaluating: checking coverage and confidence of retrieved precedents…"
                ),
            }
        )

        irac_summary = json.dumps(
            state.irac.model_dump() if state.irac else {}, ensure_ascii=False, indent=2
        )
        doc_summaries = [
            {
                "file_name": c.get("file_name"),
                "case_name": c.get("case_name"),
                "score": round(c.get("relevance_score", 0), 3),
            }
            for c in state.deduped_context
        ]
        user_content = (
            f"ORIGINAL QUERY:\n{state.query_text}\n\n"
            f"IRAC ANALYSIS:\n{irac_summary}\n\n"
            f"RETRIEVED DOCUMENTS ({len(state.deduped_context)} unique):\n"
            + json.dumps(doc_summaries, ensure_ascii=False)
            + "\n\nEvaluate and produce the reflection JSON."
        )

        try:
            await asyncio.sleep(_INTER_REQUEST_DELAY)
            resp = await self._llm.chat(
                [
                    {"role": "system", "content": REFLECTION_PROMPT},
                    {"role": "user", "content": user_content},
                ]
            )
            data = _parse_json_block(resp.content)
            if data:
                reflection = ReflectionResult.model_validate(data)
                state.reflection = reflection
                gaps_str = (
                    f" | Gaps: {', '.join(reflection.missing_aspects[:2])}"
                    if reflection.missing_aspects
                    else ""
                )
                status_str = (
                    f" | Refining with {len(reflection.refinement_queries)} new queries"
                    if reflection.needs_more_retrieval
                    else " | Ready to synthesise"
                )
                await state.emit(
                    {
                        "type": "thinking",
                        "phase": "reflection",
                        "message": (
                            f"Confidence: {reflection.confidence:.0%} | "
                            f"Quality: {reflection.reasoning_quality}" + gaps_str + status_str
                        ),
                        "confidence": reflection.confidence,
                        "reasoning_quality": reflection.reasoning_quality,
                        "missing_aspects": reflection.missing_aspects,
                        "needs_more_retrieval": reflection.needs_more_retrieval,
                        "contradictions_addressed": reflection.contradictions_addressed,
                    }
                )
                return
        except LLMUnavailableError:
            raise
        except Exception:  # noqa: BLE001
            pass

        # Fallback: assume current evidence is adequate
        state.reflection = ReflectionResult(
            confidence=0.7,
            reasoning_quality="sufficient",
            missing_aspects=[],
            needs_more_retrieval=False,
            refinement_queries=[],
            contradictions_addressed=True,
        )


# ---------------------------------------------------------------------------
# SynthesisNode
# ---------------------------------------------------------------------------


class SynthesisNode:
    """
    Phase 5 — Final answer generation.

    Research mode  : builds a structured PrecedentAnalysis JSON (using IRAC
                     precedent strengths to guide supporting/adverse split),
                     then streams a narrative summary.
    General mode   : streams a direct answer grounded in retrieved cases
                     plus IRAC pre-analysis context.

    The node writes its output to state.result.
    """

    def __init__(self, llm_provider: LLMProvider, toolbox: ResearchToolbox) -> None:
        self._llm = llm_provider
        self._toolbox = toolbox

    async def execute(self, state: AgentState) -> None:
        is_research = self._resolve_mode(state)

        await state.emit(
            {
                "type": "query_type",
                "is_research": is_research,
                "message": (
                    "Precedent research mode — building IRAC-informed supporting/adverse analysis."
                    if is_research
                    else "General query mode — generating direct answer from retrieved cases."
                ),
            }
        )

        context_block = _format_context(state.deduped_context)

        if is_research:
            structured = await self._synthesize_research(state, context_block)

            await state.emit({"type": "streaming", "message": "Streaming research narrative…"})
            analysis_json = json.dumps(structured, ensure_ascii=False, indent=2)
            chat_response = await self._stream_llm(
                state,
                [
                    {"role": "system", "content": RESEARCH_CHAT_SYSTEM_PROMPT},
                    *state.history,
                    {
                        "role": "user",
                        "content": (
                            f"Research results:\n{analysis_json}\n\nWrite the narrative summary."
                        ),
                    },
                ],
            )
            state.result = {
                "query_type": "precedent_research",
                "chat_response": chat_response,
                "response": structured,
                "sources_searched": len(state.all_retrieved),
            }
        else:
            await state.emit(
                {"type": "reasoning", "message": "LLM composing answer from retrieved cases…"}
            )
            await state.emit({"type": "streaming", "message": "Streaming answer…"})
            irac_context = _format_irac_summary(state.irac)
            chat_response = await self._stream_llm(
                state,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are Casey, an expert Indian legal research assistant. "
                            "Using ONLY the retrieved judgments and IRAC analysis provided, "
                            "answer the user's question accurately. "
                            "Cite document names and case names where relevant. "
                            "Be comprehensive and structured."
                        ),
                    },
                    *state.history,
                    {
                        "role": "user",
                        "content": (
                            f"QUERY:\n{state.query_text}\n\n"
                            + (f"IRAC PRE-ANALYSIS:\n{irac_context}\n\n" if irac_context else "")
                            + f"RETRIEVED JUDGMENTS:\n{context_block}"
                        ),
                    },
                ],
            )
            state.result = {
                "query_type": "general_query",
                "chat_response": chat_response,
                "response": self._toolbox.finalize_general_response(
                    chat_response, state.deduped_context
                ),
                "sources_searched": len(state.all_retrieved),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_mode(self, state: AgentState) -> bool:
        if state.force_mode == "research":
            return True
        if state.force_mode == "general":
            return False
        if state.plan:
            return state.plan.query_type == "precedent_research"
        return False

    async def _synthesize_research(self, state: AgentState, context_block: str) -> dict[str, Any]:
        irac_context = _format_irac_summary(state.irac)
        try:
            await asyncio.sleep(_INTER_REQUEST_DELAY)
            resp = await self._llm.chat(
                [
                    {"role": "system", "content": RESEARCH_SYNTHESIS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"USER'S CASE / QUERY:\n{state.query_text}\n\n"
                            + (f"IRAC PRE-ANALYSIS:\n{irac_context}\n\n" if irac_context else "")
                            + f"RETRIEVED JUDGMENTS:\n{context_block}\n\n"
                            "Produce the JSON analysis now."
                        ),
                    },
                ]
            )
            data = _parse_json_block(resp.content)
            if data:
                return PrecedentAnalysis.model_validate(data).model_dump()
        except LLMUnavailableError:
            raise
        except Exception:  # noqa: BLE001
            pass
        return self._fallback_research_response(state.deduped_context, state.irac)

    async def _stream_llm(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
    ) -> str:
        full_content = ""
        try:
            async for token in self._llm.chat_stream(messages):
                if token:
                    await state.emit({"type": "stream_chunk", "content": token})
                    full_content += token
            return full_content
        except Exception:  # noqa: BLE001
            try:
                resp = await self._llm.chat(messages)
                full_content = resp.content or ""
                if full_content:
                    await state.emit({"type": "stream_chunk", "content": full_content})
                return full_content
            except LLMUnavailableError:
                raise
            except Exception:  # noqa: BLE001
                return ""

    def _fallback_research_response(
        self,
        deduped: list[dict[str, Any]],
        irac: IRACReasoning | None,
    ) -> dict[str, Any]:
        """
        Build a PrecedentAnalysis from retrieved chunks when LLM synthesis fails.
        Uses IRAC precedent_strengths to split supporting (≥ 0.5) vs adverse (< 0.5).
        """
        strengths = irac.precedent_strengths if irac else {}

        supporting_docs = [
            c for c in deduped if strengths.get(c.get("document_id", ""), 0.5) >= 0.5
        ][:3]
        adverse_docs = [c for c in deduped if strengths.get(c.get("document_id", ""), 0.5) < 0.5][
            :2
        ]

        # If IRAC didn't score any docs, fall back to rank-based split
        if not supporting_docs:
            supporting_docs = deduped[:3]
        if not adverse_docs:
            adverse_docs = deduped[3:5]

        supporting = [
            SupportingPrecedent(
                document_id=c["document_id"],
                file_name=c.get("file_name", ""),
                case_name=c.get("case_name"),
                excerpt=c.get("excerpt"),
                legal_principle=(
                    irac.applicable_rules[0]
                    if irac and irac.applicable_rules
                    else "Refer to retrieved excerpt for the applicable legal rule."
                ),
                factual_alignment=(
                    irac.application
                    if irac
                    else "Case shares factual overlap with the submitted query."
                ),
            ).model_dump()
            for c in supporting_docs
        ]
        adverse = [
            AdversePrecedent(
                document_id=c["document_id"],
                file_name=c.get("file_name", ""),
                case_name=c.get("case_name"),
                excerpt=c.get("excerpt"),
                risk_description=(
                    irac.contradictions[0].description
                    if irac and irac.contradictions
                    else "This judgment may be used by the opposing party; review carefully."
                ),
                distinguishing_argument=(
                    "Distinguish on specific statutory and factual differences."
                ),
            ).model_dump()
            for c in adverse_docs
        ]
        strategy = StrategyRecommendation(
            priority_arguments=(
                irac.applicable_rules[:3]
                if irac and irac.applicable_rules
                else ["Cite the retrieved supporting judgments in primary submissions."]
            ),
            compensation_range=("Refer to the retrieved judgments for comparable award amounts."),
            risks=(
                [c.description for c in irac.contradictions[:3]]
                if irac and irac.contradictions
                else ["Review adverse precedents before proceeding to trial."]
            ),
        )
        return PrecedentAnalysis(
            supporting_precedents=supporting,
            adverse_precedents=adverse,
            strategy_recommendation=strategy,
        ).model_dump()
