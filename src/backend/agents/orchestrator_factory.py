from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.agents.enhanced_orchestrator import EnhancedOrchestrator
from src.backend.agents.intent_detector import SimpleIntentDetector
from src.backend.agents.router_service import AgentRouter
from src.backend.core.memory import MemoryManager
from src.backend.core.profile_manager import ProfileManager
from src.backend.core.response_generator import ResponseGenerator


def create_enhanced_orchestrator(db: AsyncSession) -> EnhancedOrchestrator:
    """
    Fabryka tworząca instancję EnhancedOrchestrator z wymaganymi zależnościami.

    Args:
        db: Sesja bazy danych (AsyncSession)

    Returns:
        Instancja EnhancedOrchestrator
    """
    # Utwórz menedżer profilów
    profile_manager = ProfileManager(db)

    # Utwórz detektor intencji
    intent_detector = SimpleIntentDetector()

    # Utwórz router agentów
    agent_router = AgentRouter()

    # Utwórz menedżera pamięci
    memory_manager = MemoryManager()

    # Utwórz generator odpowiedzi
    response_generator = ResponseGenerator()

    # Utwórz i zwróć orchestrator
    return EnhancedOrchestrator(
        db=db,
        profile_manager=profile_manager,
        intent_detector=intent_detector,
        agent_router=agent_router,
        memory_manager=memory_manager,
        response_generator=response_generator,
    )
