import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..core.profile_manager import ProfileManager
from ..core.sqlalchemy_compat import AsyncSession
from ..models.user_profile import InteractionType
from .orchestration_components import (
    IAgentRouter,
    IIntentDetector,
    IMemoryManager,
    IResponseGenerator,
)
from .orchestrator_errors import OrchestratorError

logger = logging.getLogger(__name__)


class EnhancedOrchestrator:
    """
    Refactored orchestrator using component-based architecture:
    - IntentDetector: Handles intent recognition
    - AgentRouter: Routes requests to appropriate agents
    - MemoryManager: Manages conversation context
    - ResponseGenerator: Creates final responses
    """

    def __init__(
        self,
        db: AsyncSession,
        profile_manager: ProfileManager,
        intent_detector: IIntentDetector,
        agent_router: IAgentRouter,
        memory_manager: IMemoryManager,
        response_generator: IResponseGenerator,
    ):
        self.db = db
        self.profile_manager = profile_manager
        self.intent_detector = intent_detector
        self.agent_router = agent_router
        self.memory_manager = memory_manager
        self.response_generator = response_generator

    def _format_error_response(self, error: Exception) -> Dict[str, Any]:
        """Format a standardized error response"""
        error_message = "Przepraszam, wystąpił błąd podczas przetwarzania żądania."
        if isinstance(error, OrchestratorError):
            error_message = str(error) or error_message

        return {
            "response": error_message,
            "status": "error",
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
        }

    async def process_file(
        self,
        file_bytes: bytes,
        filename: str,
        session_id: str,
        content_type: str,
    ) -> Dict[str, Any]:
        """
        Process an uploaded file through the orchestrator
        """
        try:
            # 1. Get user profile for personalization
            await self.profile_manager.get_or_create_profile(session_id)

            # 2. Get conversation context
            context = await self.memory_manager.get_context(session_id)

            # 3. Log file upload for analytics
            await self.profile_manager.log_activity(
                session_id, InteractionType.FILE_UPLOAD, filename
            )

            # 4. Route to appropriate agent based on file type
            if content_type.startswith("image/"):
                intent = {"type": "image_processing", "entities": {}}
            elif content_type == "application/pdf":
                intent = {"type": "document_processing", "entities": {}}
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

            # 5. Route to appropriate agent
            agent_response = await self.agent_router.route_to_agent(
                intent,
                context,
                file_data={
                    "bytes": file_bytes,
                    "filename": filename,
                    "content_type": content_type,
                },
            )

            # 6. Generate final response
            final_response = await self.response_generator.generate_response(
                agent_response, context
            )

            return {
                "response": final_response,
                "metadata": {"filename": filename, "content_type": content_type},
            }

        except OrchestratorError as e:
            logger.error(f"Orchestrator file processing error: {e}")
            return self._format_error_response(e)
        except Exception as e:
            logger.error(f"Unexpected error processing file: {e}")
            return self._format_error_response(e)

    async def process_command(
        self,
        user_command: str,
        session_id: str,
        file_info: Optional[Dict] = None,
        agent_states: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point to process a user's command using component architecture
        """
        start_time = datetime.now()

        try:
            # 1. Get user profile for personalization
            await self.profile_manager.get_or_create_profile(session_id)

            # 2. Get conversation context
            context = await self.memory_manager.get_context(session_id)

            # 3. Log user interaction for analytics
            await self.profile_manager.log_activity(
                session_id, InteractionType.QUERY, user_command
            )

            # 4. Detect intent
            intent = await self.intent_detector.detect_intent(user_command, context)

            # 5. Route to appropriate agent
            agent_response = await self.agent_router.route_to_agent(intent, context)

            # 6. Generate final response
            final_response = await self.response_generator.generate_response(
                agent_response, context
            )

            # 7. Update context with the interaction
            await self.memory_manager.update_context(
                context,
                {"user_command": user_command, "agent_response": agent_response},
            )

            # 8. Calculate and log processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Processed command in %.2fs (intent: %s)",
                processing_time,
                intent.type,
            )

            return {
                "response": final_response,
                "metadata": {
                    "processing_time": processing_time,
                    "intent": intent.type,
                    "entities": intent.entities,
                },
            }

        except OrchestratorError as e:
            logger.error(f"Orchestrator specific error: {e}")
            return self._format_error_response(e)
        except Exception as e:
            logger.error(f"Unexpected error processing command: {e}")
            return self._format_error_response(e)

    async def shutdown(self) -> None:
        """Clean shutdown of all components"""
        try:
            # Close any resources if needed
            pass
        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")


# Export for direct import will be handled elsewhere to avoid circular imports
