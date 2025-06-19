import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore

# Import orchestrator instance
from src.backend.agents.orchestrator import IntentType, Orchestrator
from src.backend.api.food import get_db

router = APIRouter()


class OrchestratorRequest(BaseModel):
    task: str
    session_id: Optional[str] = None
    conversation_state: Optional[Dict[str, Any]] = None
    agent_states: Optional[Dict[str, bool]] = None


class AgentResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Any] = None
    session_id: str
    conversation_state: Optional[Dict[str, Any]] = None


@router.post("/agents/execute", response_model=AgentResponse)
async def execute_orchestrator_task(
    request: OrchestratorRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Główny endpoint do zlecania zadań.
    Orchestrator sam decyduje, który agent wykona zadanie.
    """
    session_id = request.session_id or str(uuid.uuid4())
    try:
        # The orchestrator now manages state internally via session_id
        orchestrator = Orchestrator(db=db)

        # The process_command method handles state retrieval and message history
        response_data = await orchestrator.process_command(
            user_command=request.task,
            session_id=session_id,
            agent_states=request.agent_states,
        )

        # The state is now returned as part of the orchestrator's response
        current_state_dict = response_data.get("state", {})

        return AgentResponse(
            success=True,
            response=response_data.get("response"),
            data=response_data.get("data"),
            session_id=session_id,
            conversation_state=current_state_dict,
        )
    except Exception as e:
        # Also return session_id on error so client can continue
        return AgentResponse(success=False, error=str(e), session_id=session_id)


@router.get("/agents", response_model=List[Dict[str, str]])
async def list_available_agents():
    """Zwraca listę wszystkich dostępnych intencji."""
    return [{"name": intent.name, "description": intent.value} for intent in IntentType]
