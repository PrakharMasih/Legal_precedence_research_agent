from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from src.agent.output_schemas import (
    AdversePrecedent,
    PrecedentAnalysis,
    StrategyRecommendation,
    SupportingPrecedent,
)
from src.agent.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    RESEARCH_CHAT_SYSTEM_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
    SYSTEM_PROMPT,
)
from src.agent.tools import ResearchToolbox
from src.core.exceptions import LLMUnavailableError
from src.llm.base import LLMProvider

_MAX_ITERATIONS = 6  # max tool-calling rounds in the agentic loop
_MIN_CHUNKS_FOR_SYNTHESIS = 20  # Early exit if we have enough context
_INTER_REQUEST_DELAY = 1.0  # Delay (seconds) between LLM calls to avoid rate limits (esp. Groq)
_MAX_CONTEXT_CHUNKS = 15  # deduplicated chunks passed to synthesis LLM

# Types for streaming events and step logging
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

# Event types that are meaningful to persist (exclude granular stream_chunk)
_LOGGABLE_TYPES = frozenset(
    {
        "thinking",
        "llm_thinking",
        "tool_call",
        "tool_result",
        "synthesizing",
        "classifying",
        "query_type",
        "reasoning",
        "no_results",
        "streaming",
    }
)


class LegalResearchAgent:
    def __init__(self, *, llm_provider: LLMProvider, toolbox: ResearchToolbox) -> None:
        self._llm_provider = llm_provider
        self._toolbox = toolbox
        self._history: list[dict[str, str]] = []
        self._on_event: EventCallback | None = None
        self._steps: list[dict[str, Any]] = []  # persisted thinking trace
        self._step_counter: int = 0

    async def _emit(self, event: dict[str, Any]) -> None:
        """Attach step number, log to steps list, then forward to caller."""
        self._step_counter += 1
        event = {"step": self._step_counter, **event}
        if event.get("type") in _LOGGABLE_TYPES:
            self._steps.append(event)
        if self._on_event is not None:
            await self._on_event(event)

    async def run(
        self,
        query_text: str,
        correlation_id: str,
        history: list[dict[str, str]] | None = None,
        on_event: EventCallback | None = None,
        force_mode: str | None = None,  # "research" | "general" | None (auto)
    ) -> dict[str, Any]:
        """Run the agent. Returns result dict including `agent_steps` for DB storage."""
        self._history = history or []
        self._on_event = on_event
        self._steps = []
        self._step_counter = 0
        started_at = datetime.now(UTC)

        if self._is_conversational(query_text):
            result = await self._conversational_response(query_text)
            result["processing_time_ms"] = int(
                (datetime.now(UTC) - started_at).total_seconds() * 1000
            )
            result["agent_steps"] = self._steps
            return result

        # Phase 1 — agentic tool-calling loop: LLM decides what to search
        all_retrieved = await self._run_tool_loop(query_text)

        # Phase 2 — synthesis: deduplicate, classify, produce structured + streamed narrative
        result = await self._synthesize(query_text, all_retrieved, force_mode=force_mode)

        result["processing_time_ms"] = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
        result["agent_steps"] = self._steps
        return result

    # ------------------------------------------------------------------
    # Phase 1: Agentic tool-calling loop
    # ------------------------------------------------------------------

    async def _run_tool_loop(self, query_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": query_text},
        ]
        all_retrieved: list[dict[str, Any]] = []

        await self._emit(
            {"type": "thinking", "message": "Planning search strategy across the corpus…"}
        )

        for iteration in range(_MAX_ITERATIONS):
            # Early exit: if we have sufficient context, move to synthesis
            if len(all_retrieved) >= _MIN_CHUNKS_FOR_SYNTHESIS:
                await self._emit(
                    {
                        "type": "thinking",
                        "message": (
                            f"[Round {iteration + 1}] Sufficient context gathered "
                            f"({len(all_retrieved)} chunks). Moving to synthesis."
                        ),
                    }
                )
                break

            # Add delay before LLM call (except first iteration) to avoid rate limits
            if iteration > 0:
                await asyncio.sleep(_INTER_REQUEST_DELAY)

            await self._emit(
                {
                    "type": "llm_thinking",
                    "message": f"[Round {iteration + 1}] Deciding which searches to run…",
                }
            )
            try:
                llm_response = await self._llm_provider.chat(
                    messages, tools=self._toolbox.tool_schemas()
                )
            except LLMUnavailableError:
                raise

            if not llm_response.tool_calls:
                await self._emit(
                    {
                        "type": "thinking",
                        "message": (
                            f"[Round {iteration + 1}] LLM decided no more searches needed. "
                            f"({len(all_retrieved)} chunks gathered). Moving to synthesis."
                        ),
                    }
                )
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": llm_response.tool_calls,
                }
            )

            for tool_call in llm_response.tool_calls:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments", "{}")
                try:
                    args: dict[str, Any] = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = {}

                await self._emit({"type": "tool_call", "tool": tool_name, "args": args})

                tool_result = await self._call_tool(tool_name, args)

                if tool_name == "search_corpus" and isinstance(tool_result, list):
                    all_retrieved.extend(tool_result)
                    preview = [
                        {
                            "file_name": r.get("file_name", ""),
                            "relevance_score": round(r.get("relevance_score", 0), 3),
                            "excerpt_preview": (r.get("excerpt") or "")[:150],
                        }
                        for r in tool_result[:5]
                    ]
                    await self._emit(
                        {
                            "type": "tool_result",
                            "tool": "search_corpus",
                            "query": args.get("query", ""),
                            "total_returned": len(tool_result),
                            "top_results": preview,
                        }
                    )
                elif tool_name == "get_document_summary":
                    await self._emit(
                        {
                            "type": "tool_result",
                            "tool": "get_document_summary",
                            "document_id": args.get("document_id", ""),
                            "summary": tool_result,
                        }
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )

        return all_retrieved

    async def _call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "search_corpus":
            return await self._toolbox.search_corpus(
                query=args.get("query", ""),
                n_results=int(args.get("n_results", 10)),
                search_mode=str(args.get("search_mode", "hybrid")),
            )
        if tool_name == "get_document_summary":
            return await self._toolbox.get_document_summary(
                document_id=str(args.get("document_id", ""))
            )
        return {"error": f"Unknown tool: {tool_name}"}

    # ------------------------------------------------------------------
    # Phase 2: Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query_text: str,
        all_retrieved: list[dict[str, Any]],
        force_mode: str | None = None,
    ) -> dict[str, Any]:
        # Deduplicate: keep highest-scoring chunk per document
        seen: dict[str, dict[str, Any]] = {}
        for chunk in all_retrieved:
            doc_id = chunk.get("document_id", "")
            if not doc_id:
                continue
            if doc_id not in seen or chunk.get("relevance_score", 0) > seen[doc_id].get(
                "relevance_score", 0
            ):
                seen[doc_id] = chunk

        deduped = sorted(seen.values(), key=lambda c: c.get("relevance_score", 0), reverse=True)[
            :_MAX_CONTEXT_CHUNKS
        ]

        if not deduped:
            await self._emit(
                {
                    "type": "no_results",
                    "message": "No relevant judgments found in the indexed corpus.",
                }
            )
            await self._emit({"type": "streaming", "message": "Composing response…"})
            chat_response = await self._stream_llm(
                [
                    {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                    *self._history,
                    {"role": "user", "content": query_text},
                ]
            )
            return {
                "query_type": "general_query",
                "chat_response": chat_response,
                "response": self._toolbox.finalize_general_response(
                    "No relevant judgments found in the indexed corpus for this query.", []
                ),
                "sources_searched": 0,
            }

        await self._emit(
            {
                "type": "synthesizing",
                "message": (
                    f"Analysing {len(deduped)} unique judgment(s) "
                    f"from {len(all_retrieved)} total retrieved chunks…"
                ),
                "unique_documents": [
                    {
                        "file_name": c.get("file_name", ""),
                        "case_name": c.get("case_name"),
                        "score": round(c.get("relevance_score", 0), 3),
                    }
                    for c in deduped
                ],
            }
        )

        # Determine mode
        if force_mode == "research":
            is_research = True
        elif force_mode == "general":
            is_research = False
        else:
            await self._emit({"type": "classifying", "message": "Classifying query type…"})
            is_research = await self._llm_classify_query(query_text)

        await self._emit(
            {
                "type": "query_type",
                "is_research": is_research,
                "message": (
                    "Precedent research mode — building supporting/adverse analysis."
                    if is_research
                    else "General query mode — generating direct answer."
                ),
            }
        )

        context_block = self._format_context(deduped)

        if is_research:
            # Step A: structured JSON analysis (blocking — needs full response)
            await self._emit(
                {"type": "reasoning", "message": "LLM reasoning over retrieved precedents…"}
            )
            structured = await self._synthesize_research(query_text, context_block, deduped)

            # Step B: stream narrative summary of the analysis
            await self._emit({"type": "streaming", "message": "Streaming research narrative…"})
            analysis_json = json.dumps(structured, ensure_ascii=False, indent=2)
            chat_response = await self._stream_llm(
                [
                    {"role": "system", "content": RESEARCH_CHAT_SYSTEM_PROMPT},
                    *self._history,
                    {
                        "role": "user",
                        "content": (
                            f"Research results:\n{analysis_json}\n\nWrite the narrative summary."
                        ),
                    },
                ]
            )
            return {
                "query_type": "precedent_research",
                "chat_response": chat_response,
                "response": structured,
                "sources_searched": len(all_retrieved),
            }

        else:
            # General query: stream the answer directly
            await self._emit(
                {"type": "reasoning", "message": "LLM composing answer from retrieved cases…"}
            )
            await self._emit({"type": "streaming", "message": "Streaming answer…"})
            chat_response = await self._stream_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are Lexi, an expert Indian legal research assistant. "
                            "Using ONLY the retrieved judgments provided, answer the user's "
                            "question accurately. Cite document names and case names where "
                            "relevant. Be comprehensive and structured."
                        ),
                    },
                    *self._history,
                    {
                        "role": "user",
                        "content": (
                            f"QUERY:\n{query_text}\n\nRETRIEVED JUDGMENTS:\n{context_block}"
                        ),
                    },
                ]
            )
            return {
                "query_type": "general_query",
                "chat_response": chat_response,
                "response": self._toolbox.finalize_general_response(chat_response, deduped),
                "sources_searched": len(all_retrieved),
            }

    async def _stream_llm(self, messages: list[dict[str, Any]]) -> str:
        """Stream LLM response token-by-token, emitting stream_chunk events.

        Falls back to a single non-streaming call if streaming fails.
        Returns the full accumulated content string.
        """
        full_content = ""
        try:
            async for token in self._llm_provider.chat_stream(messages):
                if token:
                    await self._emit({"type": "stream_chunk", "content": token})
                    full_content += token
            return full_content
        except (LLMUnavailableError, Exception):  # noqa: BLE001
            # Fallback: non-streaming
            try:
                resp = await self._llm_provider.chat(messages)
                full_content = resp.content
                if full_content:
                    await self._emit({"type": "stream_chunk", "content": full_content})
                return full_content
            except LLMUnavailableError:
                raise
            except Exception:  # noqa: BLE001
                return ""

    async def _synthesize_research(
        self,
        query_text: str,
        context_block: str,
        deduped: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask LLM to produce structured PrecedentAnalysis JSON from retrieved cases."""
        try:
            resp = await self._llm_provider.chat(
                [
                    {"role": "system", "content": RESEARCH_SYNTHESIS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"USER'S CASE / QUERY:\n{query_text}\n\n"
                            f"RETRIEVED JUDGMENTS:\n{context_block}\n\n"
                            "Produce the JSON analysis now."
                        ),
                    },
                ]
            )
            return self._parse_research_json(resp.content, deduped)
        except LLMUnavailableError:
            raise
        except Exception:  # noqa: BLE001
            return self._fallback_research_response(deduped)

    def _parse_research_json(self, raw: str, deduped: list[dict[str, Any]]) -> dict[str, Any]:
        json_match = re.search(r"\{[\s\S]+\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return PrecedentAnalysis.model_validate(data).model_dump()
            except Exception:  # noqa: BLE001
                pass
        return self._fallback_research_response(deduped)

    def _fallback_research_response(self, deduped: list[dict[str, Any]]) -> dict[str, Any]:
        supporting = [
            SupportingPrecedent(
                document_id=c["document_id"],
                file_name=c.get("file_name", ""),
                case_name=c.get("case_name"),
                excerpt=c.get("excerpt"),
                legal_principle="Refer to the retrieved excerpt for the applicable legal rule.",
                factual_alignment="Case shares factual overlap with the submitted query.",
            ).model_dump()
            for c in deduped[:3]
        ]
        adverse = [
            AdversePrecedent(
                document_id=c["document_id"],
                file_name=c.get("file_name", ""),
                case_name=c.get("case_name"),
                excerpt=c.get("excerpt"),
                risk_description=(
                    "This judgment may be used by the opposing party; review carefully."
                ),
                distinguishing_argument=(
                    "Distinguish on specific statutory and factual differences."
                ),
            ).model_dump()
            for c in deduped[3:5]
        ]
        strategy = StrategyRecommendation(
            priority_arguments=["Cite the retrieved supporting judgments in primary submissions."],
            compensation_range="Refer to the retrieved judgments for comparable award amounts.",
            risks=["Review adverse precedents before proceeding to trial."],
        )
        return PrecedentAnalysis(
            supporting_precedents=supporting,
            adverse_precedents=adverse,
            strategy_recommendation=strategy,
        ).model_dump()

    async def _llm_classify_query(self, query_text: str) -> bool:
        try:
            resp = await self._llm_provider.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a legal query classifier. "
                            "Reply with exactly one word — either 'research' or 'general'.\n"
                            "'research' = the user wants precedent analysis, case strategy, "
                            "supporting/adverse precedents, litigation advice, or compensation "
                            "estimation.\n"
                            "'general' = exploratory query, factual question, document listing, "
                            "definition, or any query that does not require precedent comparison."
                        ),
                    },
                    {"role": "user", "content": query_text},
                ]
            )
            return "research" in resp.content.lower()
        except Exception:  # noqa: BLE001
            lowered = query_text.lower()
            return any(
                w in lowered
                for w in ("strategy", "precedent", "adverse", "support our case", "argue")
            )

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            case_name = chunk.get("case_name") or "Unknown"
            parts.append(
                f"[{i}] CASE: {case_name}\n"
                f"Document: {chunk.get('file_name', 'unknown')} (ID: {chunk.get('document_id', '')})\n"
                f"Relevance score: {chunk.get('relevance_score', 0):.3f}\n"
                f"Section: {chunk.get('section', 'N/A')}\n"
                f"Excerpt:\n{chunk.get('excerpt', '')}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Conversational (small-talk) path
    # ------------------------------------------------------------------

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

    def _is_conversational(self, query_text: str) -> bool:
        stripped = query_text.strip().rstrip("!?.,").lower()
        if stripped in self._CONVERSATIONAL_PATTERNS:
            return True
        words = stripped.split()
        if len(words) <= 3:
            legal_words = {
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
            return not any(w in legal_words for w in words)
        return False

    async def _conversational_response(self, query_text: str) -> dict[str, Any]:
        await self._emit({"type": "streaming", "message": "Composing conversational response…"})
        chat_response = await self._stream_llm(
            [
                {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                *self._history,
                {"role": "user", "content": query_text},
            ]
        )
        if not chat_response:
            chat_response = "Hello! How can I help you with your legal research today?"
        return {
            "query_type": "general_query",
            "chat_response": chat_response,
            "response": self._toolbox.finalize_general_response("", []),
            "sources_searched": 0,
        }
