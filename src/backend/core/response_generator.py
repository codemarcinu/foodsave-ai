from typing import Any, Dict


class ResponseGenerator:
    async def generate_response(
        self, agent_response: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """
        Generuje końcową odpowiedź na podstawie odpowiedzi agenta i kontekstu.
        W rzeczywistej implementacji powinien formatować odpowiedź i dodawać personalizację.
        """
        return agent_response.get("response", "Brak odpowiedzi od agenta.")
