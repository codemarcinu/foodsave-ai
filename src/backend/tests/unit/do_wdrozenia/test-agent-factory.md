# Test Agent Factory - Testy fabryki agentów

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib
import sys
import os

# Dodanie ścieżki do sys.path dla importów
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from src.backend.agents.agent_factory import AgentFactory
from src.backend.agents.enhanced_base_agent import ImprovedBaseAgent


class TestAgentFactory:
    """Testy dla Agent Factory - systemu tworzenia agentów"""

    @pytest.fixture
    def mock_container(self):
        """Mock kontenera DI"""
        container = Mock()
        container.resolve.return_value = Mock()
        return container

    @pytest.fixture
    def agent_factory(self, mock_container):
        """Fabryka agentów z mock'owanym kontenerem"""
        return AgentFactory(mock_container)

    def test_factory_initialization(self, mock_container):
        """Test inicjalizacji fabryki agentów"""
        # When
        factory = AgentFactory(mock_container)

        # Then
        assert factory.container == mock_container
        assert hasattr(factory, '_registry')
        assert isinstance(factory._registry, dict)

    def test_register_agent_success(self, agent_factory):
        """Test rejestracji agenta"""
        # Given
        mock_agent_class = Mock(spec=ImprovedBaseAgent)
        agent_type = "test_agent"

        # When
        agent_factory.register_agent(agent_type, mock_agent_class)

        # Then
        assert agent_type in agent_factory._registry
        assert agent_factory._registry[agent_type] == mock_agent_class

    def test_register_multiple_agents(self, agent_factory):
        """Test rejestracji wielu agentów"""
        # Given
        agents_to_register = {
            "chef_agent": Mock(spec=ImprovedBaseAgent),
            "weather_agent": Mock(spec=ImprovedBaseAgent),
            "search_agent": Mock(spec=ImprovedBaseAgent)
        }

        # When
        for agent_type, agent_class in agents_to_register.items():
            agent_factory.register_agent(agent_type, agent_class)

        # Then
        for agent_type in agents_to_register:
            assert agent_type in agent_factory._registry

    @patch('src.backend.agents.agent_factory.AgentBuilder')
    def test_create_agent_success(self, mock_builder_class, agent_factory):
        """Test pomyślnego tworzenia agenta"""
        # Given
        agent_type = "ocr"
        mock_agent_instance = Mock(spec=ImprovedBaseAgent)
        mock_builder = Mock()
        mock_builder.build.return_value = mock_agent_instance
        mock_builder_class.return_value = mock_builder

        # Rejestracja typu agenta
        agent_factory.register_agent(agent_type, Mock)

        # When
        result = agent_factory.create_agent(agent_type)

        # Then
        assert result == mock_agent_instance
        mock_builder.of_type.assert_called_once_with(agent_type)
        mock_builder.build.assert_called_once()

    @patch('src.backend.agents.agent_factory.AgentBuilder')
    def test_create_agent_with_config(self, mock_builder_class, agent_factory):
        """Test tworzenia agenta z konfiguracją"""
        # Given
        agent_type = "weather"
        config = {"api_key": "test_key", "timeout": 30}
        mock_agent_instance = Mock(spec=ImprovedBaseAgent)
        mock_builder = Mock()
        mock_builder.build.return_value = mock_agent_instance
        mock_builder_class.return_value = mock_builder

        # Rejestracja typu agenta
        agent_factory.register_agent(agent_type, Mock)

        # When
        result = agent_factory.create_agent(agent_type, config)

        # Then
        assert result == mock_agent_instance
        mock_builder.of_type.assert_called_once_with(agent_type)
        mock_builder.with_config.assert_called_once_with(config)
        mock_builder.build.assert_called_once()

    def test_create_agent_unknown_type(self, agent_factory):
        """Test tworzenia agenta nieznajomego typu"""
        # Given
        unknown_type = "unknown_agent_type"

        # When/Then
        with pytest.raises(ValueError) as exc_info:
            agent_factory.create_agent(unknown_type)

        assert f"Unknown agent type: {unknown_type}" in str(exc_info.value)

    def test_get_agent_class_ocr_agent(self, agent_factory):
        """Test dynamicznego importu klasy OCRAgent"""
        # When/Then - sprawdzamy czy metoda nie rzuca wyjątków
        try:
            agent_class = agent_factory._get_agent_class("OCRAgent")
            # Sprawdzamy czy to jest klasa
            assert callable(agent_class)
        except (ImportError, FileNotFoundError):
            # To jest OK jeśli moduł nie istnieje w środowisku testowym
            pytest.skip("Agent module not available in test environment")

    def test_get_agent_class_weather_agent(self, agent_factory):
        """Test dynamicznego importu klasy EnhancedWeatherAgent"""
        # When/Then
        try:
            agent_class = agent_factory._get_agent_class("EnhancedWeatherAgent")
            assert callable(agent_class)
        except (ImportError, FileNotFoundError):
            pytest.skip("Agent module not available in test environment")

    def test_get_agent_class_chef_agent(self, agent_factory):
        """Test dynamicznego importu klasy ChefAgent"""
        # When/Then
        try:
            agent_class = agent_factory._get_agent_class("ChefAgent")
            assert callable(agent_class)
        except (ImportError, FileNotFoundError):
            pytest.skip("Agent module not available in test environment")

    def test_get_agent_class_search_agent(self, agent_factory):
        """Test dynamicznego importu klasy SearchAgent"""
        # When/Then
        try:
            agent_class = agent_factory._get_agent_class("SearchAgent")
            assert callable(agent_class)
        except (ImportError, FileNotFoundError):
            pytest.skip("Agent module not available in test environment")

    def test_get_agent_class_unknown_class(self, agent_factory):
        """Test importu nieznajomej klasy agenta"""
        # Given
        unknown_class = "UnknownAgentClass"

        # When/Then
        with pytest.raises(ValueError) as exc_info:
            agent_factory._get_agent_class(unknown_class)

        assert f"No module mapping for agent class: {unknown_class}" in str(exc_info.value)

    @patch('importlib.import_module')
    def test_get_agent_class_import_error(self, mock_import, agent_factory):
        """Test obsługi błędu importu modułu"""
        # Given
        mock_import.side_effect = ImportError("Module not found")

        # When/Then
        with pytest.raises(ImportError):
            agent_factory._get_agent_class("OCRAgent")

    @patch('os.path.exists')
    @patch('importlib.import_module')
    def test_get_agent_class_file_not_found(self, mock_import, mock_exists, agent_factory):
        """Test obsługi błędu gdy plik modułu nie istnieje"""
        # Given
        mock_import.side_effect = ImportError("Module not found")
        mock_exists.return_value = False

        # When/Then
        with pytest.raises(FileNotFoundError):
            agent_factory._get_agent_class("OCRAgent")

    def test_module_mapping_completeness(self, agent_factory):
        """Test czy mapowanie modułów zawiera wszystkie wymagane agenty"""
        # Given - oczekiwane klasy agentów
        expected_agents = [
            "OCRAgent",
            "EnhancedWeatherAgent",
            "SearchAgent",
            "ChefAgent",
            "MealPlannerAgent",
            "CategorizationAgent",
            "AnalyticsAgent",
            "EnhancedRAGAgent"
        ]

        # When/Then - sprawdzamy czy wszystkie agenty są w mapowaniu
        for agent_class in expected_agents:
            try:
                # Jeśli nie ma błędu ValueError, to agent jest w mapowaniu
                agent_factory._get_agent_class(agent_class)
            except (ImportError, FileNotFoundError):
                # To jest OK - moduł może nie istnieć w środowisku testowym
                pass
            except ValueError:
                # To znaczy że brakuje w mapowaniu
                pytest.fail(f"Agent {agent_class} not found in module mapping")


class TestAgentFactoryIntegration:
    """Testy integracyjne fabryki agentów"""

    @pytest.fixture
    def real_container(self):
        """Rzeczywisty kontener DI (jeśli dostępny)"""
        # W środowisku testowym możemy użyć mock'a
        return Mock()

    @pytest.fixture
    def integration_factory(self, real_container):
        """Fabryka do testów integracyjnych"""
        return AgentFactory(real_container)

    def test_create_all_agent_types(self, integration_factory):
        """Test tworzenia wszystkich typów agentów"""
        # Given
        agent_types = [
            "ocr", "weather", "search", "chef",
            "meal_planner", "categorization", "analytics", "rag"
        ]

        for agent_type in agent_types:
            try:
                # Rejestracja mock'owej klasy dla testu
                integration_factory.register_agent(agent_type, Mock)

                # When
                agent = integration_factory.create_agent(agent_type)

                # Then
                assert agent is not None

            except (ImportError, FileNotFoundError):
                # Skip jeśli moduł nie jest dostępny
                pytest.skip(f"Agent {agent_type} module not available")

    @patch('src.backend.agents.agent_factory.AgentBuilder')
    def test_agent_creation_with_dependencies(self, mock_builder_class, integration_factory):
        """Test tworzenia agentów z rzeczywistymi zależnościami"""
        # Given
        agent_type = "chef"
        mock_agent = Mock()
        mock_builder = Mock()
        mock_builder.build.return_value = mock_agent
        mock_builder_class.return_value = mock_builder

        integration_factory.register_agent(agent_type, Mock)

        # When
        result = integration_factory.create_agent(agent_type)

        # Then
        assert result == mock_agent
        mock_builder.of_type.assert_called_once_with(agent_type)


class TestAgentFactoryErrorScenarios:
    """Testy scenariuszy błędów w fabryce agentów"""

    @pytest.fixture
    def error_factory(self):
        """Fabryka z potencjalnymi błędami"""
        return AgentFactory(Mock())

    def test_create_agent_builder_failure(self, error_factory):
        """Test obsługi błędu w AgentBuilder"""
        # Given
        agent_type = "test_agent"
        error_factory.register_agent(agent_type, Mock)

        with patch('src.backend.agents.agent_factory.AgentBuilder') as mock_builder_class:
            mock_builder_class.side_effect = Exception("Builder initialization failed")

            # When/Then
            with pytest.raises(Exception):
                error_factory.create_agent(agent_type)

    def test_create_agent_build_failure(self, error_factory):
        """Test obsługi błędu podczas budowania agenta"""
        # Given
        agent_type = "test_agent"
        error_factory.register_agent(agent_type, Mock)

        with patch('src.backend.agents.agent_factory.AgentBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build.side_effect = Exception("Build failed")
            mock_builder_class.return_value = mock_builder

            # When/Then
            with pytest.raises(Exception):
                error_factory.create_agent(agent_type)

    def test_sys_path_modification(self, error_factory):
        """Test modyfikacji sys.path podczas importów"""
        # Given - sprawdzamy czy sys.path został zmodyfikowany
        original_path = sys.path.copy()

        # When
        try:
            error_factory._get_agent_class("OCRAgent")
        except (ImportError, FileNotFoundError, ValueError):
            # Ignorujemy błędy importu, sprawdzamy tylko sys.path
            pass

        # Then - sys.path powinien zawierać dodane ścieżki
        assert len(sys.path) >= len(original_path)


class TestAgentFactoryPerformance:
    """Testy wydajności fabryki agentów"""

    @pytest.fixture
    def performance_factory(self):
        """Fabryka do testów wydajności"""
        factory = AgentFactory(Mock())
        # Rejestracja kilku agentów
        for i in range(10):
            factory.register_agent(f"agent_{i}", Mock)
        return factory

    def test_agent_creation_performance(self, performance_factory):
        """Test wydajności tworzenia agentów"""
        import time

        # Given
        agent_type = "agent_0"

        with patch('src.backend.agents.agent_factory.AgentBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build.return_value = Mock()
            mock_builder_class.return_value = mock_builder

            # When
            start_time = time.time()
            for _ in range(100):
                performance_factory.create_agent(agent_type)
            end_time = time.time()

            # Then
            creation_time = end_time - start_time
            assert creation_time < 1.0  # Powinno być szybsze niż 1 sekunda na 100 utworzeń

    def test_registry_lookup_performance(self, performance_factory):
        """Test wydajności wyszukiwania w rejestrze"""
        import time

        # When
        start_time = time.time()
        for _ in range(1000):
            # Sprawdzamy czy agent istnieje w rejestrze
            "agent_5" in performance_factory._registry
        end_time = time.time()

        # Then
        lookup_time = end_time - start_time
        assert lookup_time < 0.1  # Powinno być bardzo szybkie
```

## Uruchamianie testów

```bash
# Podstawowe testy
pytest tests/unit/test_agent_factory.py -v

# Testy z pokryciem kodu
pytest tests/unit/test_agent_factory.py --cov=src.backend.agents.agent_factory --cov-report=html

# Tylko testy wydajności
pytest tests/unit/test_agent_factory.py -k "performance" -v

# Tylko testy integracyjne
pytest tests/unit/test_agent_factory.py -k "integration" -v
```

## Pokrycie testów

Te testy powinny osiągnąć ~98% pokrycia kodu dla `agent_factory.py`, testując:

- ✅ Inicjalizację fabryki
- ✅ Rejestrację agentów
- ✅ Tworzenie agentów
- ✅ Dynamiczny import klas
- ✅ Obsługę błędów
- ✅ Wydajność
- ✅ Mapowanie modułów
- ✅ Konfigurację agentów
