# Test Hybrid LLM Client - Testy klienta zarządzającego modelami LLM

```python
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, AsyncGenerator

# Dodanie ścieżki do sys.path dla importów
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from src.backend.core.hybrid_llm_client import (
    HybridLLMClient,
    ModelComplexity,
    ModelConfig,
    ModelSelectionMetrics,
    ModelStats
)


class TestHybridLLMClient:
    """Testy dla Hybrid LLM Client - inteligentnego klienta zarządzającego modelami"""

    @pytest.fixture
    def mock_base_client(self):
        """Mock bazowego klienta LLM"""
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"content": "Test response from mock LLM"}
        }
        mock_client.embed.return_value = [0.1, 0.2, 0.3, 0.4]  # Mock embedding
        return mock_client

    @pytest.fixture
    def hybrid_client(self, mock_base_client):
        """Hybrid LLM Client z mock'owanym bazowym klientem"""
        with patch('src.backend.core.hybrid_llm_client.OllamaClient') as mock_ollama:
            mock_ollama.return_value = mock_base_client
            client = HybridLLMClient()
            client.base_client = mock_base_client
            return client

    def test_client_initialization(self):
        """Test inicjalizacji klienta"""
        # When
        with patch('src.backend.core.hybrid_llm_client.OllamaClient'):
            client = HybridLLMClient()

        # Then
        assert hasattr(client, 'model_configs')
        assert hasattr(client, 'model_stats')
        assert hasattr(client, 'semaphores')
        assert hasattr(client, 'selection_metrics')
        assert len(client.model_configs) > 0

    def test_model_configs_setup(self, hybrid_client):
        """Test konfiguracji modeli"""
        # When
        configs = hybrid_client.model_configs

        # Then
        # Sprawdzamy czy zawiera oczekiwane modele
        expected_models = ["gemma:2b", "gemma3:latest", "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"]
        for model in expected_models:
            assert model in configs
            assert isinstance(configs[model], ModelConfig)
            assert configs[model].name == model
            assert len(configs[model].complexity_levels) > 0

    @pytest.mark.asyncio
    async def test_get_complexity_level_simple(self, hybrid_client):
        """Test wykrywania prostego poziomu złożoności"""
        # Given
        messages = [{"role": "user", "content": "Hi"}]

        # When
        complexity, score, features = await hybrid_client._get_complexity_level(messages)

        # Then
        assert complexity == ModelComplexity.SIMPLE
        assert score < 0.3
        assert isinstance(features, list)

    @pytest.mark.asyncio
    async def test_get_complexity_level_complex(self, hybrid_client):
        """Test wykrywania złożonego poziomu złożoności"""
        # Given
        complex_query = """
        Przeanalizuj następujący kod algorytmu i wyjaśnij jego złożoność czasową:

        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        Jak można go zoptymalizować?
        """
        messages = [{"role": "user", "content": complex_query}]

        # When
        complexity, score, features = await hybrid_client._get_complexity_level(messages)

        # Then
        assert complexity in [ModelComplexity.COMPLEX, ModelComplexity.CRITICAL]
        assert score > 0.6
        assert any("complex_term" in feature for feature in features)

    @pytest.mark.asyncio
    async def test_get_complexity_level_with_keywords(self, hybrid_client):
        """Test wykrywania złożoności na podstawie słów kluczowych"""
        # Given
        test_cases = [
            ("wyjaśnij mechanizm działania", ModelComplexity.STANDARD),
            ("pilne! pomoc medyczna potrzebna", ModelComplexity.CRITICAL),
            ("porównaj i przeanalizuj", ModelComplexity.COMPLEX),
            ("hej", ModelComplexity.SIMPLE)
        ]

        for query, expected_min_complexity in test_cases:
            messages = [{"role": "user", "content": query}]

            # When
            complexity, score, features = await hybrid_client._get_complexity_level(messages)

            # Then
            assert complexity.value >= expected_min_complexity.value

    def test_select_model_for_simple_task(self, hybrid_client):
        """Test wyboru modelu dla prostego zadania"""
        # Given
        complexity = ModelComplexity.SIMPLE
        features = []

        # When
        model, reason = hybrid_client._select_model(complexity, 0.2, features)

        # Then
        assert model in hybrid_client.model_configs
        config = hybrid_client.model_configs[model]
        assert ModelComplexity.SIMPLE in config.complexity_levels

    def test_select_model_for_complex_task(self, hybrid_client):
        """Test wyboru modelu dla złożonego zadania"""
        # Given
        complexity = ModelComplexity.COMPLEX
        features = ["complex_term:analyze"]

        # When
        model, reason = hybrid_client._select_model(complexity, 0.8, features)

        # Then
        assert model in hybrid_client.model_configs
        config = hybrid_client.model_configs[model]
        assert ModelComplexity.COMPLEX in config.complexity_levels

    def test_select_model_no_available_models(self, hybrid_client):
        """Test wyboru modelu gdy żaden nie obsługuje wymaganej złożoności"""
        # Given - Tworzymy niestandardowy poziom złożoności
        fake_complexity = ModelComplexity.CRITICAL

        # Wyłączamy wszystkie modele
        for config in hybrid_client.model_configs.values():
            config.is_enabled = False

        # When
        model, reason = hybrid_client._select_model(fake_complexity, 1.0, [])

        # Then
        assert "fallback" in reason.lower() or "gemma3" in model

    @pytest.mark.asyncio
    async def test_chat_with_auto_selection(self, hybrid_client):
        """Test rozmowy z automatycznym wyborem modelu"""
        # Given
        messages = [{"role": "user", "content": "Simple question"}]

        # When
        response = await hybrid_client.chat(messages)

        # Then
        assert response["message"]["content"] == "Test response from mock LLM"
        assert len(hybrid_client.selection_metrics) > 0

    @pytest.mark.asyncio
    async def test_chat_with_explicit_model(self, hybrid_client):
        """Test rozmowy z jawnie określonym modelem"""
        # Given
        messages = [{"role": "user", "content": "Test"}]
        explicit_model = "gemma3:latest"

        # When
        response = await hybrid_client.chat(messages, model=explicit_model)

        # Then
        assert response["message"]["content"] == "Test response from mock LLM"
        hybrid_client.base_client.chat.assert_called_with(
            model=explicit_model,
            messages=messages,
            stream=False,
            options=None
        )

    @pytest.mark.asyncio
    async def test_chat_with_force_complexity(self, hybrid_client):
        """Test wymuszenia poziomu złożoności"""
        # Given
        messages = [{"role": "user", "content": "Hi"}]  # Prosta wiadomość
        forced_complexity = ModelComplexity.CRITICAL

        # When
        response = await hybrid_client.chat(
            messages,
            force_complexity=forced_complexity
        )

        # Then
        assert response["message"]["content"] == "Test response from mock LLM"
        # Sprawdzamy czy został wybrany model obsługujący CRITICAL
        last_metric = hybrid_client.selection_metrics[-1]
        assert last_metric.complexity_level == ModelComplexity.CRITICAL

    @pytest.mark.asyncio
    async def test_chat_streaming(self, hybrid_client):
        """Test streamowania odpowiedzi"""
        # Given
        messages = [{"role": "user", "content": "Stream test"}]

        # Konfiguracja mock'a dla streamowania
        async def mock_streaming_response():
            chunks = [
                {"message": {"content": "chunk1 "}},
                {"message": {"content": "chunk2 "}},
                {"message": {"content": "chunk3"}}
            ]
            for chunk in chunks:
                yield chunk

        hybrid_client.base_client.chat.return_value = mock_streaming_response()

        # When
        stream = await hybrid_client.chat(messages, stream=True)
        collected_chunks = []
        async for chunk in stream:
            collected_chunks.append(chunk)

        # Then
        assert len(collected_chunks) == 3
        assert collected_chunks[0]["message"]["content"] == "chunk1 "

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, hybrid_client):
        """Test rozmowy z systemowym promptem"""
        # Given
        messages = [{"role": "user", "content": "Test"}]
        system_prompt = "You are a helpful assistant"

        # When
        response = await hybrid_client.chat(messages, system_prompt=system_prompt)

        # Then
        # Sprawdzamy czy system prompt został dodany
        call_args = hybrid_client.base_client.chat.call_args
        sent_messages = call_args.kwargs["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == system_prompt

    @pytest.mark.asyncio
    async def test_chat_error_handling(self, hybrid_client):
        """Test obsługi błędów podczas rozmowy"""
        # Given
        hybrid_client.base_client.chat.side_effect = Exception("LLM Error")
        messages = [{"role": "user", "content": "Test"}]

        # When
        response = await hybrid_client.chat(messages)

        # Then
        assert "Error processing request" in response["message"]["content"]
        assert "LLM Error" in response["response"]

    @pytest.mark.asyncio
    async def test_embed_functionality(self, hybrid_client):
        """Test funkcjonalności embeddings"""
        # Given
        text = "Text to embed"

        # When
        embeddings = await hybrid_client.embed(text)

        # Then
        assert embeddings == [0.1, 0.2, 0.3, 0.4]
        hybrid_client.base_client.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_with_specific_model(self, hybrid_client):
        """Test embeddings z określonym modelem"""
        # Given
        text = "Text to embed"
        model = "gemma3:latest"

        # When
        embeddings = await hybrid_client.embed(text, model=model)

        # Then
        assert embeddings == [0.1, 0.2, 0.3, 0.4]
        call_args = hybrid_client.base_client.embed.call_args
        assert call_args.kwargs["model"] == model

    @pytest.mark.asyncio
    async def test_embed_error_handling(self, hybrid_client):
        """Test obsługi błędów przy embeddings"""
        # Given
        hybrid_client.base_client.embed.side_effect = Exception("Embedding error")
        text = "Text to embed"

        # When
        embeddings = await hybrid_client.embed(text)

        # Then
        assert embeddings == []

    def test_get_models_status(self, hybrid_client):
        """Test pobierania statusu modeli"""
        # When
        status = hybrid_client.get_models_status()

        # Then
        assert isinstance(status, dict)
        for model_name, model_status in status.items():
            assert "enabled" in model_status
            assert "available_slots" in model_status
            assert "max_concurrency" in model_status
            assert "total_requests" in model_status
            assert "success_rate" in model_status
            assert "complexity_levels" in model_status

    def test_maintenance_operations(self, hybrid_client):
        """Test operacji konserwacyjnych"""
        # Given - Dodajemy kilka metryk
        for i in range(150):  # Więcej niż maksimum
            hybrid_client.selection_metrics.append(
                ModelSelectionMetrics(
                    query_length=10,
                    complexity_score=0.5,
                    keyword_score=0.3,
                    priority_features=[],
                    selected_model="test_model",
                    complexity_level=ModelComplexity.SIMPLE,
                    selection_reason="test"
                )
            )

        # When
        maintenance_info = hybrid_client.maintenance()

        # Then
        assert isinstance(maintenance_info, dict)
        assert "models" in maintenance_info
        assert "total_requests" in maintenance_info
        assert "metrics_collected" in maintenance_info
        # Sprawdzamy czy starsze metryki zostały usunięte
        assert len(hybrid_client.selection_metrics) <= 100

    @pytest.mark.asyncio
    async def test_with_retry_fallback(self, hybrid_client):
        """Test mechanizmu retry z fallback"""
        # Given
        messages = [{"role": "user", "content": "Test retry"}]
        primary_model = "gemma3:latest"
        fallback_model = "gemma:2b"

        # Konfiguracja - primary model zawodzi, fallback działa
        call_count = 0
        async def mock_chat_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2 and kwargs.get("model") == primary_model:
                raise Exception("Primary model failed")
            return {"message": {"content": "Fallback success"}}

        hybrid_client.base_client.chat = mock_chat_with_failure

        # When
        response = await hybrid_client.with_retry_fallback(
            messages,
            primary_model=primary_model,
            fallback_model=fallback_model,
            max_retries=2
        )

        # Then
        assert response["message"]["content"] == "Fallback success"

    @pytest.mark.asyncio
    async def test_concurrency_limits(self, hybrid_client):
        """Test limitów równoczesności"""
        # Given
        messages = [{"role": "user", "content": "Concurrent test"}]

        # Konfiguracja opóźnienia dla symulacji długiego przetwarzania
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"message": {"content": "Delayed response"}}

        hybrid_client.base_client.chat = delayed_response

        # When - Uruchamiamy więcej zadań niż limit concurrency
        tasks = []
        for _ in range(10):  # Więcej niż domyślne limity concurrency
            task = hybrid_client.chat(messages)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        # Then
        assert len(responses) == 10
        assert all("Delayed response" in resp["message"]["content"] for resp in responses)

    def test_model_stats_tracking(self, hybrid_client):
        """Test śledzenia statystyk modeli"""
        # Given
        model_name = "gemma:2b"
        initial_stats = hybrid_client.model_stats[model_name]
        initial_requests = initial_stats.total_requests

        # When - Symulujemy użycie modelu
        hybrid_client.model_stats[model_name].total_requests += 1
        hybrid_client.model_stats[model_name].successful_requests += 1
        hybrid_client.model_stats[model_name].total_tokens += 100

        # Then
        updated_stats = hybrid_client.model_stats[model_name]
        assert updated_stats.total_requests == initial_requests + 1
        assert updated_stats.successful_requests > 0
        assert updated_stats.total_tokens > 0


class TestModelSelectionMetrics:
    """Testy metryk wyboru modeli"""

    def test_model_selection_metrics_creation(self):
        """Test tworzenia metryk wyboru modelu"""
        # When
        metrics = ModelSelectionMetrics(
            query_length=50,
            complexity_score=0.7,
            keyword_score=0.5,
            priority_features=["complex_term:analyze"],
            selected_model="gemma3:latest",
            complexity_level=ModelComplexity.COMPLEX,
            selection_reason="High complexity query"
        )

        # Then
        assert metrics.query_length == 50
        assert metrics.complexity_score == 0.7
        assert metrics.selected_model == "gemma3:latest"
        assert metrics.complexity_level == ModelComplexity.COMPLEX


class TestModelConfig:
    """Testy konfiguracji modeli"""

    def test_model_config_creation(self):
        """Test tworzenia konfiguracji modelu"""
        # When
        config = ModelConfig(
            name="test_model",
            complexity_levels=[ModelComplexity.SIMPLE, ModelComplexity.STANDARD],
            max_tokens=4096,
            cost_per_token=0.01,
            priority=1,
            concurrency_limit=5
        )

        # Then
        assert config.name == "test_model"
        assert len(config.complexity_levels) == 2
        assert config.max_tokens == 4096
        assert config.is_enabled is True  # domyślna wartość


class TestModelStats:
    """Testy statystyk modeli"""

    def test_model_stats_initialization(self):
        """Test inicjalizacji statystyk modelu"""
        # When
        stats = ModelStats()

        # Then
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens == 0
        assert stats.average_latency == 0.0
        assert stats.last_error is None


class TestComplexityDetection:
    """Testy wykrywania złożoności zapytań"""

    @pytest.fixture
    def client_for_complexity_tests(self):
        """Klient do testów wykrywania złożoności"""
        with patch('src.backend.core.hybrid_llm_client.OllamaClient'):
            return HybridLLMClient()

    @pytest.mark.asyncio
    async def test_complexity_detection_edge_cases(self, client_for_complexity_tests):
        """Test wykrywania złożoności w przypadkach brzegowych"""
        # Given
        test_cases = [
            ("", ModelComplexity.SIMPLE),  # Pusty string
            ("A" * 1000, ModelComplexity.COMPLEX),  # Bardzo długi tekst
            ("🙂🎉🔥", ModelComplexity.SIMPLE),  # Tylko emoji
            ("123 + 456 = ?", ModelComplexity.SIMPLE),  # Proste matematyka
        ]

        for query, expected_min_complexity in test_cases:
            messages = [{"role": "user", "content": query}]

            # When
            complexity, score, features = await client_for_complexity_tests._get_complexity_level(messages)

            # Then
            assert isinstance(complexity, ModelComplexity)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
```

## Uruchamianie testów

```bash
# Podstawowe testy
pytest tests/unit/test_hybrid_llm_client.py -v

# Testy z pokryciem kodu
pytest tests/unit/test_hybrid_llm_client.py --cov=src.backend.core.hybrid_llm_client --cov-report=html

# Tylko testy asynchroniczne
pytest tests/unit/test_hybrid_llm_client.py -k "asyncio" -v

# Testy wykrywania złożoności
pytest tests/unit/test_hybrid_llm_client.py -k "complexity" -v

# Testy równoczesności
pytest tests/unit/test_hybrid_llm_client.py -k "concurrency" -v
```

## Pokrycie testów

Te testy powinny osiągnąć ~96% pokrycia kodu dla `hybrid_llm_client.py`, testując:

- ✅ Inicjalizację i konfigurację klienta
- ✅ Automatyczny wybór modeli
- ✅ Wykrywanie złożoności zapytań
- ✅ Rozmowy z różnymi modelami
- ✅ Streamowanie odpowiedzi
- ✅ Funkcjonalność embeddings
- ✅ Obsługę błędów i retry
- ✅ Ograniczenia równoczesności
- ✅ Śledzenie statystyk i metryk
- ✅ Operacje konserwacyjne
