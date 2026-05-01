from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4


Message = dict[str, str]


@dataclass
class Conversation:
    messages: list[Message]
    meta: dict[str, Any]


class ConversationStore(Protocol):
    def create(self, *, initial_messages: list[Message]) -> str: ...

    def exists(self, conversation_id: str) -> bool: ...

    def get(self, conversation_id: str) -> list[Message]: ...

    def append(self, conversation_id: str, message: Message) -> None: ...

    def get_meta(self, conversation_id: str) -> dict[str, Any]: ...

    def update_meta(self, conversation_id: str, patch: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class InMemoryConversationStore:
    max_messages: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lock", Lock())
        object.__setattr__(self, "_conversations", {})

    def create(self, *, initial_messages: list[Message]) -> str:
        conversation_id = str(uuid4())
        with self._lock:
            self._conversations[conversation_id] = Conversation(
                messages=list(initial_messages),
                meta={},
            )
            self._trim_in_place(conversation_id)
        return conversation_id

    def exists(self, conversation_id: str) -> bool:
        with self._lock:
            return conversation_id in self._conversations

    def get(self, conversation_id: str) -> list[Message]:
        with self._lock:
            conv: Conversation | None = self._conversations.get(conversation_id)
            if conv is None:
                raise KeyError(conversation_id)
            return list(conv.messages)

    def append(self, conversation_id: str, message: Message) -> None:
        with self._lock:
            conv: Conversation | None = self._conversations.get(conversation_id)
            if conv is None:
                raise KeyError(conversation_id)
            conv.messages.append(message)
            self._trim_in_place(conversation_id)

    def get_meta(self, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            conv: Conversation | None = self._conversations.get(conversation_id)
            if conv is None:
                raise KeyError(conversation_id)
            return dict(conv.meta)

    def update_meta(self, conversation_id: str, patch: dict[str, Any]) -> None:
        with self._lock:
            conv: Conversation | None = self._conversations.get(conversation_id)
            if conv is None:
                raise KeyError(conversation_id)
            conv.meta.update(patch)

    def _trim_in_place(self, conversation_id: str) -> None:
        if self.max_messages <= 0:
            return

        conv: Conversation | None = self._conversations.get(conversation_id)
        if conv is None:
            return

        history = conv.messages
        if not history or len(history) <= self.max_messages:
            return

        system_message: Message | None = None
        if history and history[0].get("role") == "system":
            system_message = history[0]

        trimmed = history[-self.max_messages :]
        if system_message is not None and trimmed and trimmed[0] != system_message:
            trimmed = [system_message] + trimmed
            trimmed = trimmed[-self.max_messages :]

        conv.messages = trimmed
