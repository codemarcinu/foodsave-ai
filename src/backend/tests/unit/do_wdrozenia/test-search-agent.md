# Test Search Agent - Testy agenta wyszukiwania internetowego

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, AsyncGenerator

# Dodanie ścieżki do sys.path dla importów
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from src.backend.agents.search_agent import SearchAgent
from src.backend.agents.enhanced_base_agent import AgentResponse


class TestSearchAgent:
    """Testy dla Search Agent - agenta wyszukiwania internetowego przez DuckDuckGo"""

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock klienta HTTPX"""
        with patch('src.backend.agents.search_agent.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Konfiguracja domyślnej odpowiedzi
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "Results": [
                    {
                        "Text": "Test Result 1",
                        "FirstURL": "https://example.com/result1",
                        "Result": "Description of result 1"
                    }
                ],
                "RelatedTopics": [
                    {
                        "Text": "Related Topic 1",
                        "FirstURL": "https://example.com/related1",
                        "Result": "Description of related topic 1"
                    }
                ]
            }
            mock_client.get.return_value = mock_response

            yield mock_client

    @pytest.fixture
    def search_agent(self):
        """Fixture dla Search Agent"""
        agent = SearchAgent()
        return agent

    def test_initialization(self, search_agent):
        """Test inicjalizacji Search Agenta"""
        # Then
        assert search_agent.name == "Search Agent"
        assert search_agent.search_url == "https://api.duckduckgo.com/"

    @pytest.mark.asyncio
    async def test_process_with_valid_query(self, search_agent, mock_httpx_client):
        """Test przetwarzania poprawnego zapytania wyszukiwania"""
        # Given
        input_data = {"query": "test search query", "model": "test_model"}

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja mock'a dla LLM
            async def mock_streaming_response():
                chunks = [
                    {"message": {"content": "Formatted "}},
                    {"message": {"content": "search "}},
                    {"message": {"content": "results"}}
                ]
                for chunk in chunks:
                    yield chunk

            mock_llm.chat.return_value = mock_streaming_response()

            # When
            response = await search_agent.process(input_data)

            # Then
            assert response.success is True
            assert "query" in response.data
            assert "results" in response.data
            assert response.data["query"] == "test search query"
            assert response.text_stream is not None

    @pytest.mark.asyncio
    async def test_process_with_empty_query(self, search_agent):
        """Test przetwarzania pustego zapytania"""
        # Given
        input_data = {"query": ""}

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Brak zapytania wyszukiwania" in response.error

    @pytest.mark.asyncio
    async def test_process_with_invalid_input_type(self, search_agent):
        """Test przetwarzania nieprawidłowego typu danych wejściowych"""
        # Given
        input_data = "string instead of dict"

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Nieprawidłowy format danych wejściowych" in response.error

    @pytest.mark.asyncio
    async def test_process_with_invalid_query_type(self, search_agent):
        """Test przetwarzania nieprawidłowego typu zapytania"""
        # Given
        input_data = {"query": 123}  # Liczba zamiast string

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Nieprawidłowy typ zapytania" in response.error

    @pytest.mark.asyncio
    async def test_perform_search_success(self, search_agent, mock_httpx_client):
        """Test pomyślnego wykonania wyszukiwania"""
        # Given
        query = "test search"

        # When
        results = await search_agent._perform_search(query)

        # Then
        assert len(results) > 0
        assert "title" in results[0]
        assert "url" in results[0]
        assert "snippet" in results[0]

        # Sprawdzenie parametrów wywołania API
        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == search_agent.search_url
        assert call_args[1]["params"]["q"] == "test search"

    @pytest.mark.asyncio
    async def test_perform_search_http_error(self, search_agent, mock_httpx_client):
        """Test obsługi błędu HTTP podczas wyszukiwania"""
        # Given
        query = "test search"
        # Konfiguracja błędu HTTP
        from httpx import HTTPStatusError
        mock_httpx_client.get.side_effect = HTTPStatusError("HTTP Error", request=Mock(), response=Mock())

        # When
        results = await search_agent._perform_search(query)

        # Then
        assert results == []  # Pusta lista wyników przy błędzie

    @pytest.mark.asyncio
    async def test_perform_search_request_error(self, search_agent, mock_httpx_client):
        """Test obsługi błędu żądania (np. timeout)"""
        # Given
        query = "test search"
        # Konfiguracja błędu żądania
        from httpx import RequestError
        mock_httpx_client.get.side_effect = RequestError("Request Error", request=Mock())

        # When
        results = await search_agent._perform_search(query)

        # Then
        assert results == []  # Pusta lista wyników przy błędzie

    @pytest.mark.asyncio
    async def test_perform_search_general_error(self, search_agent, mock_httpx_client):
        """Test obsługi ogólnego błędu podczas wyszukiwania"""
        # Given
        query = "test search"
        # Konfiguracja ogólnego błędu
        mock_httpx_client.get.side_effect = Exception("General Error")

        # When
        results = await search_agent._perform_search(query)

        # Then
        assert results == []  # Pusta lista wyników przy błędzie

    @pytest.mark.asyncio
    async def test_extract_search_query(self, search_agent):
        """Test ekstraktowania zapytania wyszukiwania z wejścia użytkownika"""
        # Given
        user_input = "Proszę znajdź informacje o pogodzie w Warszawie jutro"
        model = "test_model"

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja odpowiedzi LLM
            mock_llm.chat.return_value = {
                "message": {"content": "pogoda Warszawa jutro"}
            }

            # When
            extracted_query = await search_agent._extract_search_query(user_input, model)

            # Then
            assert extracted_query == "pogoda Warszawa jutro"
            mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_search_query_error(self, search_agent):
        """Test obsługi błędu podczas ekstraktowania zapytania"""
        # Given
        user_input = "Znajdź informacje"
        model = "test_model"

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja błędu LLM
            mock_llm.chat.side_effect = Exception("LLM Error")

            # When
            extracted_query = await search_agent._extract_search_query(user_input, model)

            # Then
            assert extracted_query == user_input  # Powinno zwrócić oryginalne wejście jako fallback

    @pytest.mark.asyncio
    async def test_format_search_results(self, search_agent):
        """Test formatowania wyników wyszukiwania"""
        # Given
        query = "test search"
        results = [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "snippet": "Description 1"
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "snippet": "Description 2"
            }
        ]
        model = "test_model"

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja mock'a dla LLM
            async def mock_streaming_response():
                chunks = [
                    {"message": {"content": "Formatted "}},
                    {"message": {"content": "results "}},
                    {"message": {"content": "from search"}}
                ]
                for chunk in chunks:
                    yield chunk

            mock_llm.chat.return_value = mock_streaming_response()

            # When
            generator = search_agent._format_search_results(query, results, model)
            collected_chunks = []
            async for chunk in generator:
                collected_chunks.append(chunk)

            # Then
            assert len(collected_chunks) == 3
            assert collected_chunks[0] == "Formatted "
            assert collected_chunks[1] == "results "
            assert collected_chunks[2] == "from search"
            mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_format_search_results_error(self, search_agent):
        """Test obsługi błędu podczas formatowania wyników"""
        # Given
        query = "test search"
        results = [{"title": "Result", "url": "https://example.com", "snippet": "Description"}]
        model = "test_model"

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja błędu LLM
            mock_llm.chat.side_effect = Exception("LLM Error")

            # When/Then
            with pytest.raises(Exception):  # Oczekujemy, że błąd zostanie propagowany
                generator = search_agent._format_search_results(query, results, model)
                # Musimy skonsumować generator, żeby wywołać błąd
                async for _ in generator:
                    pass

    @pytest.mark.asyncio
    async def test_process_with_multiple_results(self, search_agent, mock_httpx_client):
        """Test przetwarzania wielu wyników wyszukiwania"""
        # Given
        input_data = {"query": "test multiple results"}

        # Konfiguracja odpowiedzi z wieloma wynikami
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "Results": [
                {"Text": "Result 1", "FirstURL": "https://example.com/1", "Result": "Desc 1"},
                {"Text": "Result 2", "FirstURL": "https://example.com/2", "Result": "Desc 2"},
                {"Text": "Result 3", "FirstURL": "https://example.com/3", "Result": "Desc 3"}
            ],
            "RelatedTopics": [
                {"Text": "Related 1", "FirstURL": "https://example.com/r1", "Result": "Related Desc 1"},
                {"Text": "Related 2", "FirstURL": "https://example.com/r2", "Result": "Related Desc 2"}
            ]
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja mock'a dla LLM
            async def mock_streaming_response():
                yield {"message": {"content": "Formatted multiple results"}}

            mock_llm.chat.return_value = mock_streaming_response()

            # When
            response = await search_agent.process(input_data)

            # Then
            assert response.success is True
            assert "results" in response.data
            assert len(response.data["results"]) == 5  # 3 główne wyniki + 2 powiązane

    @pytest.mark.asyncio
    async def test_process_with_abstract_result(self, search_agent, mock_httpx_client):
        """Test przetwarzania wyników typu AbstractText"""
        # Given
        input_data = {"query": "test abstract"}

        # Konfiguracja odpowiedzi tylko z AbstractText
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "Results": [],
            "RelatedTopics": [],
            "AbstractText": "This is an abstract text result",
            "AbstractURL": "https://example.com/abstract",
            "Heading": "Abstract Heading"
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja mock'a dla LLM
            async def mock_streaming_response():
                yield {"message": {"content": "Formatted abstract result"}}

            mock_llm.chat.return_value = mock_streaming_response()

            # When
            response = await search_agent.process(input_data)

            # Then
            assert response.success is True
            assert "results" in response.data
            assert len(response.data["results"]) == 1
            assert response.data["results"][0]["title"] == "Abstract Heading"
            assert response.data["results"][0]["snippet"] == "This is an abstract text result"

    @pytest.mark.asyncio
    async def test_process_with_no_results(self, search_agent, mock_httpx_client):
        """Test przetwarzania braku wyników wyszukiwania"""
        # Given
        input_data = {"query": "no results query"}

        # Konfiguracja pustej odpowiedzi
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "Results": [],
            "RelatedTopics": []
        }
        mock_httpx_client.get.return_value = mock_response

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is True
        assert f"Brak wyników dla zapytania: {input_data['query']}" in response.text

    @pytest.mark.asyncio
    async def test_web_search_with_weather_query(self, search_agent, mock_httpx_client):
        """Test wyszukiwania informacji o pogodzie"""
        # Given
        input_data = {"query": "pogoda Warszawa", "model": "test_model"}

        # Konfiguracja odpowiedzi związanej z pogodą
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "Results": [
                {
                    "Text": "Pogoda Warszawa",
                    "FirstURL": "https://pogoda.example.com/warszawa",
                    "Result": "Aktualna pogoda w Warszawie: 22°C, słonecznie"
                }
            ],
            "RelatedTopics": [
                {
                    "Text": "Prognoza długoterminowa",
                    "FirstURL": "https://pogoda.example.com/warszawa/długoterminowa",
                    "Result": "Prognoza na 7 dni dla Warszawy"
                }
            ]
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('src.backend.agents.search_agent.llm_client') as mock_llm:
            # Konfiguracja mock'a dla LLM
            async def mock_streaming_response():
                yield {"message": {"content": "W Warszawie jest obecnie 22°C i słonecznie."}}

            mock_llm.chat.return_value = mock_streaming_response()

            # When
            response = await search_agent.process(input_data)

            # Then
            assert response.success is True
            assert "results" in response.data
            assert len(response.data["results"]) == 2
            assert "Warszawa" in response.data["results"][0]["title"]
            assert "pogoda" in response.data["results"][0]["title"].lower()
```

## Uruchamianie testów

```bash
# Podstawowe testy
pytest tests/unit/test_search_agent.py -v

# Testy z pokryciem kodu
pytest tests/unit/test_search_agent.py --cov=src.backend.agents.search_agent --cov-report=html

# Tylko testy przetwarzania zapytań
pytest tests/unit/test_search_agent.py -k "process" -v

# Testy obsługi błędów
pytest tests/unit/test_search_agent.py -k "error" -v

# Testy związane z pogodą
pytest tests/unit/test_search_agent.py -k "weather" -v
```

## Pokrycie testów

Te testy powinny osiągnąć ~95% pokrycia kodu dla `search_agent.py`, testując:

- ✅ Inicjalizację agenta wyszukiwania
- ✅ Przetwarzanie różnych zapytań
- ✅ Wykonywanie wyszukiwania DuckDuckGo
- ✅ Ekstraktowanie zapytań z wejścia użytkownika
- ✅ Formatowanie wyników wyszukiwania
- ✅ Obsługę różnych formatów odpowiedzi
- ✅ Obsługę błędów na różnych poziomach
- ✅ Wyszukiwania związane z pogodą
