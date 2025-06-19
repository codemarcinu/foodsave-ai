from typing import Any, Dict


class AgentRouter:
    async def route_to_agent(
        self, intent: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Uproszczony router agentów.
        W rzeczywistej implementacji powinien wybierać odpowiedniego agenta na podstawie intencji.
        """
        return {
            "response": "To jest domyślna odpowiedź agenta. W rzeczywistej implementacji powinien tu być wybrany właściwy agent.",
            "metadata": {
                "intent": intent.get("type", "unknown"),
                "agent": "default_agent",
            },
        }
