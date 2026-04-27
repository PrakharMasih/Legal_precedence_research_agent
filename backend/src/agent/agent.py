"""
LegalResearchAgent — public entry point for the autonomous legal reasoning pipeline.

Thin wrapper that:
1. Fast-paths obviously conversational queries (pattern-match, zero LLM calls).
2. Delegates all substantive legal queries to GraphWorkflow, which runs:
       PlannerNode → RetrievalNode → ReasonerNode → ReflectorNode → SynthesisNode
3. Stitches processing_time_ms and agent_steps onto the result dict before returning.

The public `run()` signature is unchanged; every caller (REST route + WebSocket) continues
to work without modification.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from src.agent.graph.workflow import GraphWorkflow
from src.agent.prompts import GENERAL_CHAT_SYSTEM_PROMPT
from src.agent.tools import ResearchToolbox
from src.core.exceptions import LLMUnavailableError
from src.llm.base import LLMProvider

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

# ── Conversational fast-path patterns ────────────────────────────────────────
_CONVERSATIONAL_PATTERNS: frozenset[str] = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hiya",
        "howdy",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "how are you",
        "how r u",
        "how are u",
        "what's up",
        "whats up",
        "sup",
        "who are you",
        "what are you",
        "what can you do",
        "help",
        "thanks",
        "thank you",
        "thank u",
        "ty",
        "bye",
        "goodbye",
        "ok",
        "okay",
        "cool",
        "great",
        "awesome",
        "nice",
        "got it",
        "understood",
    }
)

_LEGAL_WORDS: frozenset[str] = frozenset(
    {
        "case",
        "court",
        "judgment",
        "law",
        "legal",
        "liable",
        "liability",
        "compensation",
        "negligence",
        "accident",
        "insurance",
        "plaintiff",
        "defendant",
        "appeal",
        "tribunal",
        "damages",
        "verdict",
        "precedent",
    }
)


class LegalResearchAgent:
    """
    Autonomous legal reasoning agent.

    Pipeline (per query):

        [Fast path]  conversational pattern → direct LLM reply

        [Main path]
            Planner   → decomposes query into sub_queries + legal_issues
            Retrieval → multi-query hybrid corpus search (deterministic)
            Reasoner  → IRAC: issue / rules / application / conclusion
                         + precedent strength scores + contradiction detection
            Reflector → confidence scoring + gap analysis
                         (loops back to Retrieval if confidence < 0.6)
            Synthesis → IRAC-informed PrecedentAnalysis JSON
                         + streamed narrative response
    """

    def __init__(self, *, llm_provider: LLMProvider, toolbox: ResearchToolbox) -> None:
        self._llm_provider = llm_provider
        self._toolbox = toolbox
        self._workflow = GraphWorkflow(llm_provider, toolbox)

    async def run(
        self,
        query_text: str,
        correlation_id: str,
        history: list[dict[str, str]] | None = None,
        on_event: EventCallback | None = None,
        force_mode: str | None = None,  # "research" | "general" | None (auto)
    ) -> dict[str, Any]:
        """
        Run the agent. Returns a result dict with:
            query_type, chat_response, response,
            sources_searched, processing_time_ms, agent_steps
        """
        started_at = datetime.now(UTC)

        # ── Zero-cost conversational fast path ─────────────────────────────
        if self._is_conversational(query_text):
            result = await self._conversational_response(query_text, history or [], on_event)
        else:
            result = await self._workflow.run(
                query_text=query_text,
                correlation_id=correlation_id,
                history=history,
                on_event=on_event,
                force_mode=force_mode,
            )

        result["processing_time_ms"] = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
        result["agent_steps"] = result.pop("_steps", [])
        return result

    # ------------------------------------------------------------------
    # Conversational fast path
    # ------------------------------------------------------------------

    @staticmethod
    def _is_conversational(query_text: str) -> bool:
        """Return True for greetings / small-talk that need no corpus retrieval."""
        stripped = query_text.strip().rstrip("!?.,").lower()
        if stripped in _CONVERSATIONAL_PATTERNS:
            return True
        words = stripped.split()
        if len(words) <= 3:
            return not any(w in _LEGAL_WORDS for w in words)
        return False

    async def _conversational_response(
        self,
        query_text: str,
        history: list[dict[str, Any]],
        on_event: EventCallback | None,
    ) -> dict[str, Any]:
        step_counter = 0

        async def emit(event: dict[str, Any]) -> None:
            nonlocal step_counter
            step_counter += 1
            event = {"step": step_counter, **event}
            if on_event is not None:
                await on_event(event)

        await emit({"type": "streaming", "message": "Composing conversational response…"})

        full_content = ""
        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": query_text},
            ]
            async for token in self._llm_provider.chat_stream(messages):
                if token:
                    await emit({"type": "stream_chunk", "content": token})
                    full_content += token
        except Exception:  # noqa: BLE001
            try:
                resp = await self._llm_provider.chat(
                    [
                        {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                        *history,
                        {"role": "user", "content": query_text},
                    ]
                )
                full_content = resp.content or ""
                if full_content:
                    await emit({"type": "stream_chunk", "content": full_content})
            except LLMUnavailableError:
                raise
            except Exception:  # noqa: BLE001
                full_content = "Hello! How can I help you with your legal research today?"

        return {
            "query_type": "general_query",
            "chat_response": full_content,
            "response": self._toolbox.finalize_general_response("", []),
            "sources_searched": 0,
            "_steps": [],
        }
