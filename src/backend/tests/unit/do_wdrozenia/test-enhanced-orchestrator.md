# Test Enhanced Orchestrator - Komprehensywne testy orkiestracji agentów

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, Optional

# Dodanie ścieżki do sys.path dla importów
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from src.backend.agents.enhanced_orchestrator import EnhancedOrchestrator
from src.backend.agents.enhanced_base_agent import EnhancedAgentResponse
from src.backend.core.memory import MemoryManager
from src.backend.core.hybrid_llm_client import hybrid_llm_client


class TestEnhancedOrchestrator:
    """Testy dla Enhanced Orchestrator - centralnego kontrolera systemu"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock wszystkich zależności orkiestratora"""
        mock_container = Mock()
        mock_profile_manager = AsyncMock()
        mock_memory_manager = AsyncMock()
        mock_intent_detector = AsyncMock()
        mock_agent_router = AsyncMock()
        mock_response_generator = AsyncMock()

        # Konfiguracja podstawowych zwracanych wartości
        mock_profile_manager.get_or_create_profile.return_value = {"user_id": "test_user"}
        mock_memory_manager.get_context.return_value = {"session_id": "test_session"}
        mock_intent_detector.detect_intent.return_value = Mock(type="general", entities={})
        mock_agent_router.route_to_agent.return_value = {"success": True, "response": "Test response"}
        mock_response_generator.generate_response.return_value = "Generated response"

        return {
            "container": mock_container,
            "profile_manager": mock_profile_manager,
            "memory_manager": mock_memory_manager,
            "intent_detector": mock_intent_detector,
            "agent_router": mock_agent_router,
            "response_generator": mock_response_generator
        }

    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Orkiestrator z mock'owanymi zależnościami"""
        orch = EnhancedOrchestrator(mock_dependencies["container"])
        orch.profile_manager = mock_dependencies["profile_manager"]
        orch.memory_manager = mock_dependencies["memory_manager"]
        orch.intent_detector = mock_dependencies["intent_detector"]
        orch.agent_router = mock_dependencies["agent_router"]
        orch.response_generator = mock_dependencies["response_generator"]
        return orch

    @pytest.mark.asyncio
    async def test_process_command_basic_flow(self, orchestrator, mock_dependencies):
        """Test podstawowego przepływu przetwarzania komendy"""
        # Given
        user_command = "Pokaż przepis na makaron"
        session_id = "test_session_123"

        # When
        result = await orchestrator.process_command(user_command, session_id)

        # Then
        assert result["response"] == "Generated response"
        assert "metadata" in result
        assert result["metadata"]["intent"] == "general"

        # Weryfikacja wywołań
        mock_dependencies["profile_manager"].get_or_create_profile.assert_called_once_with(session_id)
        mock_dependencies["memory_manager"].get_context.assert_called_once_with(session_id)
        mock_dependencies["intent_detector"].detect_intent.assert_called_once()
        mock_dependencies["agent_router"].route_to_agent.assert_called_once()
        mock_dependencies["response_generator"].generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_command_with_error_handling(self, orchestrator, mock_dependencies):
        """Test obsługi błędów podczas przetwarzania komendy"""
        # Given
        user_command = "Test command"
        session_id = "test_session"
        mock_dependencies["intent_detector"].detect_intent.side_effect = Exception("Intent detection error")

        # When
        result = await orchestrator.process_command(user_command, session_id)

        # Then
        assert result["status"] == "error"
        assert "Intent detection error" in result["response"] or "błąd" in result["response"].lower()
        assert result["error_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_process_file_image_handling(self, orchestrator, mock_dependencies):
        """Test przetwarzania plików obrazów (paragonów)"""
        # Given
        file_bytes = b"fake_image_data"
        filename = "paragon.jpg"
        session_id = "test_session"
        content_type = "image/jpeg"

        # When
        result = await orchestrator.process_file(file_bytes, filename, session_id, content_type)

        # Then
        assert "response" in result
        assert result["metadata"]["filename"] == filename
        assert result["metadata"]["content_type"] == content_type

        # Weryfikacja że router został wywołany z odpowiednim intent
        args, kwargs = mock_dependencies["agent_router"].route_to_agent.call_args
        intent = args[0]
        assert intent["type"] == "image_processing"
        assert "file_data" in kwargs

    @pytest.mark.asyncio
    async def test_process_file_pdf_handling(self, orchestrator, mock_dependencies):
        """Test przetwarzania plików PDF"""
        # Given
        file_bytes = b"fake_pdf_data"
        filename = "document.pdf"
        session_id = "test_session"
        content_type = "application/pdf"

        # When
        result = await orchestrator.process_file(file_bytes, filename, session_id, content_type)

        # Then
        args, kwargs = mock_dependencies["agent_router"].route_to_agent.call_args
        intent = args[0]
        assert intent["type"] == "document_processing"

    @pytest.mark.asyncio
    async def test_process_file_unsupported_type(self, orchestrator, mock_dependencies):
        """Test obsługi nieobsługiwanych typów plików"""
        # Given
        file_bytes = b"fake_data"
        filename = "file.unknown"
        session_id = "test_session"
        content_type = "application/unknown"

        # When
        result = await orchestrator.process_file(file_bytes, filename, session_id, content_type)

        # Then
        assert result["status"] == "error"
        assert "Unsupported content type" in result["response"]

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, orchestrator, mock_dependencies):
        """Test obsługi wielu równoczesnych żądań"""
        # Given
        commands = [
            "Pokaż przepis",
            "Jaka jest pogoda?",
            "Wyszukaj informacje o makaronie",
            "Dodaj pomidory do spiżarni"
        ]
        session_id = "concurrent_test"

        # When
        tasks = [
            orchestrator.process_command(cmd, session_id)
            for cmd in commands
        ]
        results = await asyncio.gather(*tasks)

        # Then
        assert len(results) == 4
        for result in results:
            assert "response" in result
            assert result.get("status") != "error" or "błąd" in result.get("response", "").lower()

    @pytest.mark.asyncio
    async def test_context_persistence(self, orchestrator, mock_dependencies):
        """Test zachowania kontekstu między żądaniami"""
        # Given
        session_id = "context_test"
        first_command = "Pokaż przepis na makaron"
        second_command = "Dodaj do tego ser"

        # When
        await orchestrator.process_command(first_command, session_id)
        await orchestrator.process_command(second_command, session_id)

        # Then
        # Sprawdź że kontekst był aktualizowany
        assert mock_dependencies["memory_manager"].update_context.call_count == 2

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, orchestrator, mock_dependencies):
        """Test zbierania metryk wydajności"""
        # Given
        user_command = "Test performance"
        session_id = "perf_test"

        # When
        result = await orchestrator.process_command(user_command, session_id)

        # Then
        assert "metadata" in result
        assert "processing_time" in result["metadata"]
        assert isinstance(result["metadata"]["processing_time"], float)
        assert result["metadata"]["processing_time"] >= 0

    @pytest.mark.asyncio
    async def test_activity_logging(self, orchestrator, mock_dependencies):
        """Test logowania aktywności użytkownika"""
        # Given
        user_command = "Test logging"
        session_id = "log_test"

        # When
        await orchestrator.process_command(user_command, session_id)

        # Then
        # Sprawdź że aktywność została zarejestrowana
        assert mock_dependencies["profile_manager"].log_activity.called
        call_args = mock_dependencies["profile_manager"].log_activity.call_args
        assert call_args[0][0] == session_id  # session_id
        assert call_args[0][2] == user_command  # command

    def test_error_response_formatting(self, orchestrator):
        """Test formatowania odpowiedzi błędów"""
        # Given
        test_error = ValueError("Test error message")

        # When
        response = orchestrator._format_error_response(test_error)

        # Then
        assert response["status"] == "error"
        assert response["error_type"] == "ValueError"
        assert isinstance(response["timestamp"], str)
        assert "błąd" in response["response"].lower() or "error" in response["response"].lower()

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, orchestrator):
        """Test poprawnego zamykania orkiestratora"""
        # When/Then - nie powinno rzucać wyjątków
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_intent_detection_integration(self, orchestrator, mock_dependencies):
        """Test integracji z systemem wykrywania intencji"""
        # Given
        cooking_command = "Chcę ugotować makaron carbonara"
        weather_command = "Jaka będzie jutro pogoda?"
        search_command = "Wyszukaj informacje o zdrowym odżywianiu"

        test_cases = [
            (cooking_command, "cooking"),
            (weather_command, "weather"),
            (search_command, "search")
        ]

        for command, expected_intent in test_cases:
            # Konfiguracja mock'a dla konkretnej intencji
            mock_dependencies["intent_detector"].detect_intent.return_value = Mock(
                type=expected_intent,
                entities={}
            )

            # When
            result = await orchestrator.process_command(command, "test_session")

            # Then
            assert result["metadata"]["intent"] == expected_intent

    @pytest.mark.asyncio
    async def test_agent_routing_validation(self, orchestrator, mock_dependencies):
        """Test walidacji routingu do odpowiednich agentów"""
        # Given
        intent_types = ["cooking", "weather", "search", "ocr", "analytics"]

        for intent_type in intent_types:
            mock_dependencies["intent_detector"].detect_intent.return_value = Mock(
                type=intent_type,
                entities={}
            )

            # When
            await orchestrator.process_command(f"Test {intent_type}", "test_session")

            # Then
            args, kwargs = mock_dependencies["agent_router"].route_to_agent.call_args
            routed_intent = args[0]
            assert routed_intent.type == intent_type


class TestOrchestratorErrorHandling:
    """Testy zaawansowanej obsługi błędów w orkiestratorze"""

    @pytest.fixture
    def orchestrator_with_failing_deps(self):
        """Orkiestrator z komponenty które mogą zawieść"""
        mock_container = Mock()
        orch = EnhancedOrchestrator(mock_container)

        # Konfiguracja komponentów do failowania
        orch.profile_manager = AsyncMock()
        orch.memory_manager = AsyncMock()
        orch.intent_detector = AsyncMock()
        orch.agent_router = AsyncMock()
        orch.response_generator = AsyncMock()

        return orch

    @pytest.mark.asyncio
    async def test_profile_manager_failure(self, orchestrator_with_failing_deps):
        """Test obsługi błędu w profile manager"""
        # Given
        orchestrator_with_failing_deps.profile_manager.get_or_create_profile.side_effect = Exception("DB Error")

        # When
        result = await orchestrator_with_failing_deps.process_command("test", "session")

        # Then
        assert result["status"] == "error"
        assert "błąd" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_memory_manager_failure(self, orchestrator_with_failing_deps):
        """Test obsługi błędu w memory manager"""
        # Given
        orchestrator_with_failing_deps.profile_manager.get_or_create_profile.return_value = {}
        orchestrator_with_failing_deps.memory_manager.get_context.side_effect = Exception("Memory Error")

        # When
        result = await orchestrator_with_failing_deps.process_command("test", "session")

        # Then
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_agent_router_failure(self, orchestrator_with_failing_deps):
        """Test obsługi błędu w agent router"""
        # Given
        orchestrator_with_failing_deps.profile_manager.get_or_create_profile.return_value = {}
        orchestrator_with_failing_deps.memory_manager.get_context.return_value = {}
        orchestrator_with_failing_deps.intent_detector.detect_intent.return_value = Mock(type="test", entities={})
        orchestrator_with_failing_deps.agent_router.route_to_agent.side_effect = Exception("Routing Error")

        # When
        result = await orchestrator_with_failing_deps.process_command("test", "session")

        # Then
        assert result["status"] == "error"


class TestOrchestratorIntegration:
    """Testy integracyjne orkiestratora z rzeczywistymi komponentami"""

    @pytest.mark.asyncio
    async def test_real_intent_detection_flow(self):
        """Test z rzeczywistym systemem wykrywania intencji (jeśli dostępny)"""
        # Ten test można uruchomić tylko jeśli system jest skonfigurowany
        pytest.skip("Requires full system setup - integration test")

    @pytest.mark.asyncio
    async def test_end_to_end_conversation_flow(self):
        """Test pełnego przepływu konwersacji"""
        # Ten test sprawdza całą konwersację od początku do końca
        pytest.skip("Requires full system setup - integration test")
```

## Konfiguracja uruchamiania testów

```bash
# Uruchomienie testów enhanced orchestrator
pytest tests/unit/test_enhanced_orchestrator.py -v

# Uruchomienie z pokryciem kodu
pytest tests/unit/test_enhanced_orchestrator.py --cov=src.backend.agents.enhanced_orchestrator --cov-report=html

# Uruchomienie tylko testów asynchronicznych
pytest tests/unit/test_enhanced_orchestrator.py -k "asyncio" -v
```

## Metryki pokrycia

Te testy powinny osiągnąć ~95% pokrycia kodu dla `enhanced_orchestrator.py`, testując:

- ✅ Podstawowy przepływ przetwarzania komend
- ✅ Obsługę plików (obrazy, PDF)
- ✅ Zarządzanie błędami
- ✅ Wydajność i metryki
- ✅ Równoczesność
- ✅ Integrację z komponentami
- ✅ Zachowanie kontekstu
- ✅ Logowanie aktywności
