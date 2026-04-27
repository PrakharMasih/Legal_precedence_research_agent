"""
GraphWorkflow — autonomous legal reasoning state machine.

Full pipeline:

    User Query
        ↓
    [1] PlannerNode          → decomposes query into sub_queries + legal_issues
        ↓
    [2] RetrievalNode        → deterministic multi-query corpus search
        ↓
    [3] ReasonerNode (IRAC)  → issue / rules / application / conclusion
                               + precedent strength scoring
                               + contradiction detection
        ↓
    [4] ReflectorNode        → confidence score + gap analysis
        ↓
      ┌─── confidence < 0.6 AND loop budget remains?
      │         ↓ yes
      │      update sub_queries → back to [2]
      │
      └─── no
            ↓
    [5] SynthesisNode        → IRAC-informed PrecedentAnalysis JSON
                               + streamed narrative response
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from src.agent.graph.nodes import (
    PlannerNode,
    ReasonerNode,
    ReflectorNode,
    RetrievalNode,
    SynthesisNode,
)
from src.agent.graph.state import AgentState
from src.agent.prompts import GENERAL_CHAT_SYSTEM_PROMPT, GENERAL_QUERY_ANSWER_PROMPT
from src.agent.tools import ResearchToolbox
from src.llm.base import LLMProvider

_MAX_REFLECTION_LOOPS = 2  # max retrieval → reason → reflect cycles

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class GraphWorkflow:
    """
    Orchestrates the five nodes into a stateful, reflective pipeline.

    The caller receives:
        {
          "query_type":         str,
          "chat_response":      str,
          "response":           dict,        # PrecedentAnalysis or GeneralQueryResponse
          "sources_searched":   int,
          "_steps":             list[dict],  # persisted thinking trace
        }
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        toolbox: ResearchToolbox,
    ) -> None:
        self._planner = PlannerNode(llm_provider)
        self._retriever = RetrievalNode(toolbox)
        self._reasoner = ReasonerNode(llm_provider)
        self._reflector = ReflectorNode(llm_provider)
        self._synthesizer = SynthesisNode(llm_provider, toolbox)
        self._toolbox = toolbox

    async def run(
        self,
        query_text: str,
        correlation_id: str,
        history: list[dict[str, Any]] | None = None,
        on_event: EventCallback | None = None,
        force_mode: str | None = None,
    ) -> dict[str, Any]:
        state = AgentState(
            query_text=query_text,
            correlation_id=correlation_id,
            history=history or [],
            force_mode=force_mode,
            on_event=on_event,
        )

        # ── Phase 1: Plan ────────────────────────────────────────────────────
        await self._planner.execute(state)
        plan = state.plan
        assert plan is not None  # planner always writes a fallback

        # Pure conversational fast path (greetings / small talk — no corpus involvement)
        if plan.strategy == "conversational":
            return self._attach_steps(await self._conversational_response(state), state)

        # General informational query — retrieve if needed, then answer directly; skip IRAC
        if plan.query_type == "general_query":
            return self._attach_steps(await self._general_query_response(state), state)

        # ── Phases 2–4: Retrieval → IRAC Reason → Reflect (with loop) ────────
        for loop_idx in range(_MAX_REFLECTION_LOOPS):
            await self._retriever.execute(state)

            if not state.deduped_context:
                return self._attach_steps(await self._no_results_response(state), state)

            await self._reasoner.execute(state)
            await self._reflector.execute(state)

            reflection = state.reflection
            should_loop = (
                reflection is not None
                and reflection.needs_more_retrieval
                and bool(reflection.refinement_queries)
                and loop_idx < _MAX_REFLECTION_LOOPS - 1
            )
            if should_loop:
                assert reflection is not None
                await state.emit(
                    {
                        "type": "thinking",
                        "message": (
                            f"[Loop {loop_idx + 1}] Confidence {reflection.confidence:.0%} — "
                            f"refining with {len(reflection.refinement_queries)} new queries…"
                        ),
                    }
                )
                plan.sub_queries = reflection.refinement_queries
                state.retrieval_iteration += 1
            else:
                break

        # ── Phase 5: Synthesise ───────────────────────────────────────────────
        confidence_pct = f"{state.reflection.confidence:.0%}" if state.reflection else "N/A"
        await state.emit(
            {
                "type": "synthesizing",
                "message": (
                    f"Synthesising from {len(state.deduped_context)} unique judgment(s) "
                    f"(confidence: {confidence_pct})"
                ),
                "unique_documents": [
                    {
                        "file_name": c.get("file_name", ""),
                        "case_name": c.get("case_name"),
                        "score": round(c.get("relevance_score", 0), 3),
                    }
                    for c in state.deduped_context
                ],
            }
        )
        await self._synthesizer.execute(state)
        return self._attach_steps(state.result or {}, state)

    # ------------------------------------------------------------------
    # Internal response helpers
    # ------------------------------------------------------------------

    async def _general_query_response(self, state: AgentState) -> dict[str, Any]:
        """
        Lightweight path for informational / general queries.

        requires_retrieval=true  → search corpus first, then answer directly.
        requires_retrieval=false → answer from conversation history alone (follow-up
                                   referencing already-shown results); no corpus call.

        No IRAC, no reflection, no precedent analysis in either case.
        """
        plan = state.plan

        if plan and plan.requires_retrieval:
            await self._retriever.execute(state)
            if not state.deduped_context:
                return await self._no_results_response(state)

        # Build context block from retrieved chunks when available
        context_text = ""
        if state.deduped_context:
            ctx_parts: list[str] = []
            for i, chunk in enumerate(state.deduped_context, 1):
                name = chunk.get("case_name") or chunk.get("file_name", "Unknown")
                ctx_parts.append(f"[{i}] {name}\n{chunk.get('excerpt', '')}")
            context_text = "\n\n---\n\n".join(ctx_parts)

        # When retrieval was skipped the conversation history already contains the cases;
        # pass the raw question so the LLM reads them from the history messages.
        user_content = (
            f"Question: {state.query_text}\n\nRetrieved judgments:\n{context_text}"
            if context_text
            else state.query_text
        )

        await state.emit({"type": "streaming", "message": "Composing answer…"})
        chat_response = await self._synthesizer._stream_llm(
            state,
            [
                {"role": "system", "content": GENERAL_QUERY_ANSWER_PROMPT},
                *state.history,
                {"role": "user", "content": user_content},
            ],
        )
        if not chat_response:
            chat_response = "I couldn't find a clear answer in the indexed corpus."
        return {
            "query_type": "general_query",
            "chat_response": chat_response,
            "response": self._toolbox.finalize_general_response(
                chat_response, state.deduped_context
            ),
            "sources_searched": len(state.all_retrieved),
        }

    async def _conversational_response(self, state: AgentState) -> dict[str, Any]:
        await state.emit({"type": "streaming", "message": "Composing conversational response…"})
        chat_response = await self._synthesizer._stream_llm(
            state,
            [
                {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                *state.history,
                {"role": "user", "content": state.query_text},
            ],
        )
        if not chat_response:
            chat_response = "Hello! How can I help you with your legal research today?"
        return {
            "query_type": "general_query",
            "chat_response": chat_response,
            "response": self._toolbox.finalize_general_response("", []),
            "sources_searched": 0,
        }

    async def _no_results_response(self, state: AgentState) -> dict[str, Any]:
        await state.emit(
            {
                "type": "no_results",
                "message": "No relevant judgments found in the indexed corpus.",
            }
        )
        await state.emit({"type": "streaming", "message": "Composing response…"})
        chat_response = await self._synthesizer._stream_llm(
            state,
            [
                {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                *state.history,
                {"role": "user", "content": state.query_text},
            ],
        )
        return {
            "query_type": "general_query",
            "chat_response": chat_response,
            "response": self._toolbox.finalize_general_response(
                "No relevant judgments found in the indexed corpus for this query.", []
            ),
            "sources_searched": 0,
        }

    @staticmethod
    def _attach_steps(result: dict[str, Any], state: AgentState) -> dict[str, Any]:
        result["_steps"] = state.steps
        return result
