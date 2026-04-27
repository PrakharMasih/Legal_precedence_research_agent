"""Service layer for chat operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.conversation import Message

if TYPE_CHECKING:
    from src.storage.repositories import ChatRepository


class ChatService:
    """
    Service for chat history management.

    Provides a business logic layer above ChatRepository.
    """

    def __init__(self, chat_repository: ChatRepository) -> None:
        """
        Initialize ChatService.

        Args:
            chat_repository: Repository for chat persistence.
        """
        self._chat_repository = chat_repository

    async def get_chat_history(self, limit: int = 50, offset: int = 0) -> tuple[list[Message], int]:
        """
        Get paginated chat history.

        Args:
            limit: Number of messages to retrieve (1-200).
            offset: Number of messages to skip (>= 0).

        Returns:
            Tuple of (messages, total_count).
        """
        messages = await self._chat_repository.get_history(limit=limit, offset=offset)
        total = await self._chat_repository.count()
        return messages, total

    async def append_message(self, message: Message) -> None:
        """
        Store a message in chat history.

        Args:
            message: Message to store.
        """
        await self._chat_repository.append(message)

    async def get_recent_context(self) -> list[Message]:
        """
        Get recent messages for conversation context.

        Used for providing chat history to the LLM.

        Returns:
            List of recent messages.
        """
        return await self._chat_repository.get_recent_context()

    async def clear_history(self) -> None:
        """Clear all chat history."""
        await self._chat_repository.clear()
