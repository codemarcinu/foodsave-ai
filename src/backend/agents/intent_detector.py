from typing import Any, Dict


class SimpleIntentDetector:
    async def detect_intent(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uproszczony detektor intencji.
        W rzeczywistej implementacji powinien używać modelu NLP.
        """
        return {"type": "general", "entities": {}, "confidence": 1.0}
