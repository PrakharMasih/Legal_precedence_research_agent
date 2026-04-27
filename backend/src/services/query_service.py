"""Service layer for query execution and research operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.agent.agent import LegalResearchAgent
from src.agent.tools import ResearchToolbox
from src.models.conversation import Message
from src.retrieval.retriever import Retriever
from src.utils.timestamps import get_utc_now

if TYPE_CHECKING:
    from src.llm.base import LLMProvider
    from src.storage.repositories import (
        ChatRepository,
        ChunkRepository,
        DocumentRepository,
    )
    from src.storage.vector_store import VectorStore


class QueryService:
    """
    Service for executing legal research queries.

    Orchestrates:
    - Agent initialization and execution
    - Query history persistence
    - Error handling and recovery
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        chat_repository: ChatRepository,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        vector_store: VectorStore,
        embedder: Any,
    ) -> None:
        """
        Initialize QueryService.

        Args:
            llm_provider: LLM provider for text generation.
            chat_repository: Repository for storing chat history.
            document_repository: Repository for retrieving documents.
            chunk_repository: Repository for retrieving document chunks.
            vector_store: Vector store for semantic search.
            embedder: Embedder for generating embeddings.
        """
        self._llm_provider = llm_provider
        self._chat_repository = chat_repository
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._vector_store = vector_store
        self._embedder = embedder

    def _build_agent(self) -> LegalResearchAgent:
        """
        Build a configured LegalResearchAgent.

        Returns:
            Initialized agent ready for query execution.
        """
        retriever = Retriever(
            session_factory=None,  # Will be injected via runtime
            vector_store=self._vector_store,
            embedder=self._embedder,
        )
        toolbox = ResearchToolbox(
            retriever=retriever,
            document_repository=self._document_repository,
            chunk_repository=self._chunk_repository,
        )
        return LegalResearchAgent(llm_provider=self._llm_provider, toolbox=toolbox)

    async def execute_query(
        self,
        query_text: str,
        correlation_id: str,
        history: list[dict[str, str]] | None = None,
        force_mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a legal research query.

        Args:
            query_text: User's research question.
            correlation_id: Request correlation ID for tracing.
            history: Previous conversation messages for context.
            force_mode: Force "research" or "general" mode ("auto" for automatic detection).

        Returns:
            Result dict with:
                - query_type: Type of query detected
                - chat_response: Narrative response to user
                - response: Detailed research findings
                - sources_searched: Number of sources retrieved
                - processing_time_ms: Total execution time
                - agent_steps: Detailed step-by-step execution log
                - user_message_id: ID of stored user message
                - assistant_message_id: ID of stored assistant response

        Raises:
            LLMUnavailableError: If LLM provider is unreachable.
            CorpusNotIndexedError: If corpus has not been indexed yet.
        """
        agent = self._build_agent()

        result = await agent.run(
            query_text=query_text,
            correlation_id=correlation_id,
            history=history,
            force_mode=force_mode,
        )

        # Generate message IDs
        user_message_id = str(uuid4())
        assistant_message_id = str(uuid4())

        # Store messages in history
        now = get_utc_now()

        user_msg = Message(
            id=user_message_id,
            role="user",
            content=query_text,
            query_type=result.get("query_type"),
            sources_searched=result.get("sources_searched"),
            created_at=now,
        )

        assistant_msg = Message(
            id=assistant_message_id,
            role="assistant",
            content=result.get("chat_response", ""),
            query_type=result.get("query_type"),
            sources_searched=result.get("sources_searched"),
            raw_response=result.get("response"),
            agent_steps=result.get("agent_steps"),
            created_at=get_utc_now(),
        )

        await self._chat_repository.append(user_msg)
        await self._chat_repository.append(assistant_msg)

        result["user_message_id"] = user_message_id
        result["assistant_message_id"] = assistant_message_id

        return result

    async def get_recent_context(self) -> list[dict[str, str]]:
        """
        Get recent messages for providing conversation context to the agent.

        Returns:
            List of recent messages in {'role': str, 'content': str} format.
        """
        messages = await self._chat_repository.get_recent_context()
        return [{"role": m.role, "content": m.content} for m in messages]
