"""AgentState: mutable context object threaded through every graph node."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.agent.output_schemas import IRACReasoning, PlannerOutput, ReflectionResult

# Event types persisted to the DB / forwarded to WebSocket clients
_LOGGABLE_TYPES: frozenset[str] = frozenset(
    {
        # thinking covers planning / retrieval / reflection phases (distinguished by "phase" field)
        "thinking",
        "llm_thinking",
        "tool_call",
        "tool_result",
        "reasoning",
        "synthesizing",
        "classifying",
        "query_type",
        "no_results",
        "streaming",
    }
)

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class AgentState:
    # ── Inputs ────────────────────────────────────────────────────────────────
    query_text: str
    correlation_id: str
    history: list[dict[str, Any]]
    force_mode: str | None
    on_event: EventCallback | None = None

    # ── Planner output ────────────────────────────────────────────────────────
    plan: PlannerOutput | None = None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    all_retrieved: list[dict[str, Any]] = field(default_factory=list)
    deduped_context: list[dict[str, Any]] = field(default_factory=list)
    retrieval_iteration: int = 0

    # ── Reasoning ─────────────────────────────────────────────────────────────
    irac: IRACReasoning | None = None

    # ── Reflection ────────────────────────────────────────────────────────────
    reflection: ReflectionResult | None = None

    # ── Final result (populated by SynthesisNode) ─────────────────────────────
    result: dict[str, Any] | None = None

    # ── Telemetry ─────────────────────────────────────────────────────────────
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_counter: int = 0

    async def emit(self, event: dict[str, Any]) -> None:
        """Stamp event with step number, persist if loggable, forward to caller."""
        self.step_counter += 1
        event = {"step": self.step_counter, **event}
        if event.get("type") in _LOGGABLE_TYPES:
            self.steps.append(event)
        if self.on_event is not None:
            await self.on_event(event)
