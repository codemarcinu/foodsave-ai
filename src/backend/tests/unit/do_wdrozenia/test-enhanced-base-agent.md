# Test Enhanced Base Agent - Testy klasy bazowej agentów

```python
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Dodanie ścieżki do sys.path dla importów
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from src.backend.agents.enhanced_base_agent import (
    ImprovedBaseAgent,
    EnhancedAgentResponse,
    ErrorSeverity,
    AgentResponse
)


class TestEnhancedBaseAgent:
    """Testy dla Enhanced Base Agent - klasy bazowej wszystkich agentów"""

    class ConcreteTestAgent(ImprovedBaseAgent):
        """Konkretna implementacja agenta do testów"""

        def __init__(self, name="TestAgent"):
            super().__init__(name=name)
            self.process_calls = []
            self.should_fail = False
            self.response_message = "Test response"

        async def process(self, input_data: Dict[str, Any]) -> EnhancedAgentResponse:
            """Implementacja process dla testów"""
            self.process_calls.append(input_data)

            if self.should_fail:
                raise ValueError("Simulated agent failure")

            return EnhancedAgentResponse(
                success=True,
                text=self.response_message,
                message="Process completed successfully",
                metadata={"processed": True}
            )

    @pytest.fixture
    def test_agent(self):
        """Agent testowy"""
        return self.ConcreteTestAgent()

    @pytest.fixture
    def mock_dependencies(self):
        """Mock'owane zależności agenta"""
        return {
            "error_handler": Mock(),
            "fallback_manager": Mock(),
            "alert_service": Mock(),
            "llm_client": AsyncMock()
        }

    def test_agent_initialization(self):
        """Test inicjalizacji agenta"""
        # When
        agent = self.ConcreteTestAgent("CustomName")

        # Then
        assert agent.name == "CustomName"
        assert agent.fallback_attempts == 0
        assert hasattr(agent, 'error_handler')
        assert hasattr(agent, 'fallback_manager')
        assert hasattr(agent, 'alert_service')

    def test_agent_default_name(self):
        """Test domyślnej nazwy agenta"""
        # When
        agent = self.ConcreteTestAgent()

        # Then
        assert agent.name == "TestAgent"

    @pytest.mark.asyncio
    async def test_safe_process_success(self, test_agent):
        """Test pomyślnego przetwarzania przez safe_process"""
        # Given
        input_data = {"query": "test query", "param": "value"}

        # When
        response = await test_agent.safe_process(input_data)

        # Then
        assert response.success is True
        assert response.text == "Test response"
        assert response.message == "Process completed successfully"
        assert response.metadata["processed"] is True
        assert response.processing_time > 0
        assert len(test_agent.process_calls) == 1
        assert test_agent.process_calls[0] == input_data

    @pytest.mark.asyncio
    async def test_safe_process_with_failure(self, test_agent):
        """Test obsługi błędu w safe_process"""
        # Given
        test_agent.should_fail = True
        test_agent.fallback_manager = AsyncMock()
        test_agent.fallback_manager.execute_fallback.return_value = EnhancedAgentResponse(
            success=True,
            text="Fallback response",
            processed_with_fallback=True
        )
        input_data = {"query": "test query"}

        # When
        response = await test_agent.safe_process(input_data)

        # Then
        assert response.processed_with_fallback is True
        assert response.processing_time > 0
        test_agent.fallback_manager.execute_fallback.assert_called_once()

    def test_validate_input_no_model(self, test_agent):
        """Test walidacji danych wejściowych bez modelu"""
        # Given
        input_data = {"key": "value", "number": 42}

        # When
        validated = test_agent._validate_input(input_data)

        # Then
        assert validated == input_data

    def test_validate_input_with_model(self, test_agent):
        """Test walidacji danych wejściowych z modelem Pydantic"""
        # Given
        from pydantic import BaseModel

        class TestInputModel(BaseModel):
            query: str
            optional_param: int = 10

        test_agent.input_model = TestInputModel
        input_data = {"query": "test", "optional_param": 5}

        # When
        validated = test_agent._validate_input(input_data)

        # Then
        assert validated["query"] == "test"
        assert validated["optional_param"] == 5

    def test_validate_input_validation_error(self, test_agent):
        """Test błędu walidacji danych wejściowych"""
        # Given
        from pydantic import BaseModel, ValidationError

        class TestInputModel(BaseModel):
            required_field: str

        test_agent.input_model = TestInputModel
        input_data = {"wrong_field": "value"}

        # When/Then
        with pytest.raises(ValueError) as exc_info:
            test_agent._validate_input(input_data)

        assert "Invalid input data for TestAgent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, test_agent):
        """Test execute_with_fallback przy pomyślnym wykonaniu"""
        # Given
        test_agent.error_handler = AsyncMock()
        test_agent.error_handler.execute_with_fallback.return_value = "success_result"

        async def test_function():
            return "test_result"

        # When
        result = await test_agent.execute_with_fallback(test_function)

        # Then
        test_agent.error_handler.execute_with_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_tiered_fallback(self, test_agent):
        """Test wielopoziomowego systemu fallback"""
        # Given
        original_error = ValueError("Original error")
        input_data = {"query": "test"}
        start_time = time.time()

        test_agent.fallback_manager = AsyncMock()
        test_agent.alert_service = Mock()
        test_agent.alert_service.should_alert.return_value = True
        test_agent.alert_service.send_alert = AsyncMock()

        fallback_response = EnhancedAgentResponse(
            success=True,
            text="Fallback worked",
            processed_with_fallback=True
        )
        test_agent.fallback_manager.execute_fallback.return_value = fallback_response

        # When
        response = await test_agent._multi_tiered_fallback(input_data, original_error, start_time)

        # Then
        assert response.processed_with_fallback is True
        assert response.processing_time > 0
        test_agent.alert_service.send_alert.assert_called_once()
        test_agent.fallback_manager.execute_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_try_prompt_rewriting(self, test_agent):
        """Test przepisywania promptów przy błędach"""
        # Given
        input_data = {"query": "Complex query that caused error"}
        original_error = ValueError("LLM parsing error")

        # Mock hybrid_llm_client
        with patch('src.backend.agents.enhanced_base_agent.hybrid_llm_client') as mock_llm:
            mock_llm.chat.return_value = {
                "message": {"content": "Rewritten simple query"}
            }

            # Mock the process method to succeed with rewritten query
            original_process = test_agent.process
            async def mock_process(data):
                if data.get("query") == "Rewritten simple query":
                    return EnhancedAgentResponse(success=True, text="Success with rewritten query")
                else:
                    raise original_error

            test_agent.process = mock_process

            # When
            result = await test_agent._try_prompt_rewriting(input_data, original_error)

            # Then
            assert result is not None
            assert result.success is True
            assert result.metadata["original_query"] == "Complex query that caused error"
            assert result.metadata["rewritten_query"] == "Rewritten simple query"

    @pytest.mark.asyncio
    async def test_try_simplified_model(self, test_agent):
        """Test użycia uproszczonego modelu przy błędach"""
        # Given
        input_data = {"query": "Complex technical query"}
        original_error = ValueError("Model complexity error")

        # Mock hybrid_llm_client
        with patch('src.backend.agents.enhanced_base_agent.hybrid_llm_client') as mock_llm:
            mock_llm.chat.return_value = {
                "message": {"content": "Simple answer from lightweight model"}
            }

            # When
            result = await test_agent._try_simplified_model(input_data, original_error)

            # Then
            assert result is not None
            assert result.success is True
            assert result.metadata["simplified"] is True
            assert result.metadata["original_query"] == "Complex technical query"
            assert "Simple answer from lightweight model" in result.text

    @pytest.mark.asyncio
    async def test_stream_llm_response_success(self, test_agent):
        """Test streamowania odpowiedzi LLM"""
        # Given
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]

        # Mock hybrid_llm_client
        with patch('src.backend.agents.enhanced_base_agent.hybrid_llm_client') as mock_llm:
            # Symulacja streamowania
            async def mock_stream():
                chunks = [
                    {"message": {"content": "chunk1 "}},
                    {"message": {"content": "chunk2 "}},
                    {"message": {"content": "chunk3"}}
                ]
                for chunk in chunks:
                    yield chunk

            mock_llm.chat.return_value = mock_stream()

            # When
            collected_chunks = []
            async for chunk in test_agent._stream_llm_response(model, messages):
                collected_chunks.append(chunk)

            # Then
            assert len(collected_chunks) == 3
            assert collected_chunks == ["chunk1 ", "chunk2 ", "chunk3"]

    @pytest.mark.asyncio
    async def test_stream_llm_response_with_retries(self, test_agent):
        """Test streamowania z ponowami prób"""
        # Given
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]

        with patch('src.backend.agents.enhanced_base_agent.hybrid_llm_client') as mock_llm:
            # Pierwsza próba się nie udaje, druga się udaje
            call_count = 0
            async def mock_chat_with_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Network error")
                else:
                    # Sukces za drugim razem
                    async def success_stream():
                        yield {"message": {"content": "success"}}
                    return success_stream()

            mock_llm.chat = mock_chat_with_failure

            # When
            collected_chunks = []
            async for chunk in test_agent._stream_llm_response(model, messages, retries=3):
                collected_chunks.append(chunk)

            # Then
            assert len(collected_chunks) == 1
            assert collected_chunks[0] == "success"

    @pytest.mark.asyncio
    async def test_stream_llm_response_all_retries_fail(self, test_agent):
        """Test gdy wszystkie próby streamowania się nie udają"""
        # Given
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]

        with patch('src.backend.agents.enhanced_base_agent.hybrid_llm_client') as mock_llm:
            mock_llm.chat.side_effect = Exception("Persistent error")

            # When
            collected_chunks = []
            async for chunk in test_agent._stream_llm_response(model, messages, retries=2):
                collected_chunks.append(chunk)

            # Then
            assert len(collected_chunks) == 1
            assert "Error: Failed to generate response after 2 attempts" in collected_chunks[0]


class TestEnhancedAgentResponse:
    """Testy dla klasy EnhancedAgentResponse"""

    def test_enhanced_agent_response_creation(self):
        """Test tworzenia odpowiedzi agenta"""
        # When
        response = EnhancedAgentResponse(
            success=True,
            text="Test response",
            message="Success message",
            metadata={"key": "value"},
            processing_time=1.5
        )

        # Then
        assert response.success is True
        assert response.text == "Test response"
        assert response.message == "Success message"
        assert response.metadata == {"key": "value"}
        assert response.processing_time == 1.5
        assert response.processed_with_fallback is False  # domyślna wartość

    def test_enhanced_agent_response_defaults(self):
        """Test domyślnych wartości w odpowiedzi agenta"""
        # When
        response = EnhancedAgentResponse(success=True)

        # Then
        assert response.success is True
        assert response.text is None
        assert response.message is None
        assert response.metadata == {}
        assert response.processing_time is None
        assert response.processed_with_fallback is False

    def test_enhanced_agent_response_with_error(self):
        """Test odpowiedzi agenta z błędem"""
        # When
        response = EnhancedAgentResponse(
            success=False,
            error="Test error message",
            error_severity="high"
        )

        # Then
        assert response.success is False
        assert response.error == "Test error message"
        assert response.error_severity == "high"


class TestAgentErrorHandling:
    """Testy obsługi błędów w agentach"""

    @pytest.fixture
    def failing_agent(self):
        """Agent, który zawsze rzuca błędy"""

        class FailingAgent(ImprovedBaseAgent):
            async def process(self, input_data: Dict[str, Any]) -> EnhancedAgentResponse:
                raise RuntimeError("Agent always fails")

        return FailingAgent("FailingAgent")

    @pytest.mark.asyncio
    async def test_error_handling_without_fallback_manager(self, failing_agent):
        """Test obsługi błędów gdy brak fallback managera"""
        # Given
        failing_agent.fallback_manager = None
        input_data = {"test": "data"}

        # When/Then
        with pytest.raises(ValueError) as exc_info:
            await failing_agent.safe_process(input_data)

        assert "Fallback manager must be initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_critical_error_alerting(self, failing_agent):
        """Test alertowania o krytycznych błędach"""
        # Given
        failing_agent.fallback_manager = AsyncMock()
        failing_agent.alert_service = Mock()
        failing_agent.alert_service.should_alert.return_value = True
        failing_agent.alert_service.send_alert = AsyncMock()

        fallback_response = EnhancedAgentResponse(
            success=True,
            text="Fallback response",
            processed_with_fallback=True
        )
        failing_agent.fallback_manager.execute_fallback.return_value = fallback_response

        # When
        await failing_agent.safe_process({"test": "data"})

        # Then
        failing_agent.alert_service.send_alert.assert_called_once()
        call_args = failing_agent.alert_service.send_alert.call_args
        assert "Critical error in FailingAgent" in call_args[0][0]
        assert call_args[0][2] == ErrorSeverity.HIGH


class TestAgentPerformance:
    """Testy wydajności agentów"""

    @pytest.fixture
    def performance_agent(self):
        """Agent do testów wydajności"""

        class PerformanceAgent(ImprovedBaseAgent):
            def __init__(self):
                super().__init__("PerformanceAgent")
                self.call_count = 0

            async def process(self, input_data: Dict[str, Any]) -> EnhancedAgentResponse:
                self.call_count += 1
                # Symulacja pewnej ilości pracy
                await asyncio.sleep(0.01)
                return EnhancedAgentResponse(
                    success=True,
                    text=f"Response {self.call_count}"
                )

        return PerformanceAgent()

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, performance_agent):
        """Test równoczesnego przetwarzania przez agenta"""
        # Given
        tasks_count = 10
        input_data = {"test": "concurrent"}

        # When
        start_time = time.time()
        tasks = [
            performance_agent.safe_process(input_data)
            for _ in range(tasks_count)
        ]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Then
        assert len(results) == tasks_count
        assert all(result.success for result in results)
        assert performance_agent.call_count == tasks_count

        # Powinno być szybsze niż sekwencyjne wykonanie
        total_time = end_time - start_time
        assert total_time < (tasks_count * 0.01 * 0.8)  # Z pewnym marginesem

    @pytest.mark.asyncio
    async def test_processing_time_measurement(self, performance_agent):
        """Test pomiaru czasu przetwarzania"""
        # Given
        input_data = {"test": "timing"}

        # When
        response = await performance_agent.safe_process(input_data)

        # Then
        assert response.processing_time is not None
        assert response.processing_time > 0
        assert response.processing_time >= 0.01  # Minimum sleep time
```

## Uruchamianie testów

```bash
# Podstawowe testy
pytest tests/unit/test_enhanced_base_agent.py -v

# Testy z pokryciem kodu
pytest tests/unit/test_enhanced_base_agent.py --cov=src.backend.agents.enhanced_base_agent --cov-report=html

# Tylko testy asynchroniczne
pytest tests/unit/test_enhanced_base_agent.py -k "asyncio" -v

# Testy wydajności
pytest tests/unit/test_enhanced_base_agent.py -k "performance" -v

# Testy obsługi błędów
pytest tests/unit/test_enhanced_base_agent.py -k "error" -v
```

## Pokrycie testów

Te testy powinny osiągnąć ~97% pokrycia kodu dla `enhanced_base_agent.py`, testując:

- ✅ Inicjalizację agenta
- ✅ Walidację danych wejściowych
- ✅ Przetwarzanie bezpieczne (safe_process)
- ✅ Wielopoziomowy system fallback
- ✅ Przepisywanie promptów
- ✅ Uproszczone modele
- ✅ Streamowanie odpowiedzi LLM
- ✅ Obsługę błędów i alertów
- ✅ Wydajność i równoczesność
- ✅ Pomiar czasu przetwarzania
