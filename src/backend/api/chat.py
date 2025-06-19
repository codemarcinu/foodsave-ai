from typing import Any, Dict, cast

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.agents.orchestrator_factory import create_enhanced_orchestrator
from src.backend.infrastructure.database.database import get_db

from ..config import settings
from ..core.llm_client import llm_client

# APIRouter działa jak "mini-aplikacja" FastAPI, grupując endpointy
router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None  # Pole opcjonalne


class MemoryChatRequest(BaseModel):
    message: str
    session_id: str  # Kluczowe do śledzenia i odróżniania konwersacji


class MemoryChatResponse(BaseModel):
    reply: str
    history_length: int


async def chat_response_generator(prompt: str, model: str):
    """
    Asynchroniczny generator. Pobiera kawałki odpowiedzi od klienta
    Ollama i od razu przesyła je dalej (yield).
    """
    try:
        async for chunk in llm_client.generate_stream(
            model=model, prompt=prompt, system_prompt=""
        ):
            if not isinstance(chunk, dict):
                continue
            chunk_dict = cast(Dict[str, Any], chunk)
            if "response" in chunk_dict:
                yield chunk_dict["response"]
    except Exception as e:
        print(f"Błąd podczas streamowania: {e}")
        yield "Wystąpił błąd serwera."


@router.post("/chat")
async def chat_with_model(request: ChatRequest):
    """
    Endpoint do prowadzenia rozmowy z modelem LLM.
    Dzięki StreamingResponse, odpowiedź jest wysyłana do klienta
    w czasie rzeczywistym, bez czekania na całą odpowiedź modelu.
    """
    # Jeśli w zapytaniu nie podano modelu, użyj domyślnego z konfiguracji
    model_to_use = request.model or settings.DEFAULT_CODE_MODEL
    generator = chat_response_generator(request.prompt, model_to_use)
    return StreamingResponse(generator, media_type="text/plain")


async def memory_chat_generator(request: MemoryChatRequest, db: AsyncSession):
    """
    Generator for streaming responses from the orchestrator.
    """
    try:
        # Użyj fabryki do utworzenia EnhancedOrchestrator
        orchestrator = create_enhanced_orchestrator(db)

        # The orchestrator now returns a dictionary or a streaming generator
        response_data = await orchestrator.process_command(
            user_command=request.message, session_id=request.session_id
        )

        # Check if the response is a streaming generator
        if hasattr(response_data, "__aiter__"):
            async for chunk in response_data:
                yield chunk
        # If it's a regular dictionary response, yield it as a single chunk
        elif isinstance(response_data, dict):
            import json

            yield json.dumps(response_data)

    except Exception as e:
        import json

        error_response = {
            "response": f"Wystąpił błąd serwera: {str(e)}",
            "state": {},
        }
        yield json.dumps(error_response)


@router.post("/memory_chat")
async def chat_with_memory(
    request: MemoryChatRequest, db: AsyncSession = Depends(get_db)
):
    """
    Endpoint for conversing with an agent that remembers conversation history.
    Each conversation is identified by a session_id.
    Supports streaming responses.
    """
    generator = memory_chat_generator(request, db)
    return StreamingResponse(generator, media_type="application/x-ndjson")
