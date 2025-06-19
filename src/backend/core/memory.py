from typing import Any, Dict


class MemoryManager:
    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Pobiera kontekst konwersacji na podstawie identyfikatora sesji.
        W rzeczywistej implementacji powinien pobierać z bazy danych.
        """
        return {"session_id": session_id, "history": [], "user_preferences": {}}

    async def update_context(
        self, context: Dict[str, Any], updates: Dict[str, Any]
    ) -> None:
        """
        Aktualizuje kontekst konwersacji.
        W rzeczywistej implementacji powinien zapisywać zmiany w bazie danych.
        """
        context.update(updates)
