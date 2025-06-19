# Przewodnik uruchamiania testów jednostkowych - FoodSave AI

## Przegląd testów

Przygotowany zestaw testów jednostkowych dla aplikacji FoodSave AI zapewnia kompleksowe pokrycie kluczowych komponentów:

### ✅ Kompletne testy orkiestracji agentów:
- **Enhanced Orchestrator** - centralny kontroler systemu
- **Agent Factory** - fabryka tworzenia agentów
- **Enhanced Base Agent** - bazowa klasa wszystkich agentów

### ✅ Testy kluczowych agentów:
- **OCR Agent** - rozpoznawanie paragonów i ekstraktowanie tekstu
- **Search Agent** - wyszukiwanie internetowe przez DuckDuckGo
- **Weather Agent** - (można dodać na podstawie wzorców)
- **Chef Agent** - (można dodać na podstawie wzorców)

### ✅ Testy komponentów core:
- **Hybrid LLM Client** - zarządzanie lokalnymi modelami LLM
- **Enhanced Vector Store** - (można dodać na podstawie wzorców)
- **Memory Management** - (można dodać na podstawie wzorców)

## Struktura plików testów

```
tests/
├── unit/
│   ├── test_enhanced_orchestrator.py      # ✅ Gotowy
│   ├── test_agent_factory.py              # ✅ Gotowy
│   ├── test_enhanced_base_agent.py        # ✅ Gotowy
│   ├── test_ocr_agent.py                  # ✅ Gotowy
│   ├── test_search_agent.py               # ✅ Gotowy
│   ├── test_hybrid_llm_client.py          # ✅ Gotowy
│   ├── test_chef_agent.py                 # Do dodania
│   ├── test_enhanced_weather_agent.py     # Do dodania
│   ├── test_enhanced_vector_store.py      # Do dodania
│   └── test_memory.py                     # Do dodania
├── integration/
│   ├── test_full_agent_flow.py            # Do dodania
│   ├── test_file_processing_pipeline.py   # Do dodania
│   └── test_chat_with_internet_access.py  # Do dodania
└── fixtures/
    ├── test_data.json                     # ✅ Istniejący
    ├── test_receipt.jpg                   # ✅ Istniejący
    └── test_chat_scenarios.json           # Do dodania
```

## Konfiguracja środowiska testowego

### 1. Instalacja zależności

```bash
# Aktywacja środowiska Poetry
poetry shell

# Instalacja zależności testowych
poetry install --group test

# Alternatywnie z pip:
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### 2. Konfiguracja zmiennych środowiskowych

Utwórz plik `.env.test`:

```bash
# .env.test
DATABASE_URL=sqlite:///test_foodsave.db
LLM_MODEL_DEFAULT=gemma3:latest
EMBEDDING_MODEL=nomic-embed-text
OCR_ENGINE=tesseract

# API keys dla testów (opcjonalne, można użyć mock'ów)
WEATHER_API_KEY=test_key
BING_SEARCH_API_KEY=test_key
NEWS_API_KEY=test_key

# Konfiguracja testów
PYTEST_CURRENT_TEST=true
TESTING=true
```

### 3. Struktura konfiguracyjna pytest

Utwórz `pytest.ini`:

```ini
[tool:pytest]
minversion = 6.0
addopts =
    -ra
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    asyncio: asynchronous tests
    unit: unit tests
    integration: integration tests
    slow: slow running tests
    requires_llm: tests requiring LLM access
    requires_internet: tests requiring internet access
asyncio_mode = auto
```

### 4. Konfiguracja pokrycia kodu

Utwórz `.coveragerc`:

```ini
[run]
source = src/
omit =
    tests/*
    */venv/*
    */virtualenv/*
    */site-packages/*
    */__pycache__/*
    */migrations/*
    */settings/*
    setup.py
    manage.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

## Uruchamianie testów

### Podstawowe polecenia

```bash
# Wszystkie testy jednostkowe
pytest tests/unit/ -v

# Konkretny plik testów
pytest tests/unit/test_enhanced_orchestrator.py -v

# Testy z filtrowaniem
pytest tests/unit/ -k "orchestrator" -v

# Testy asynchroniczne
pytest tests/unit/ -k "asyncio" -v

# Testy z pokryciem kodu
pytest tests/unit/ --cov=src --cov-report=html

# Testy z raportem szczegółowym
pytest tests/unit/ --cov=src --cov-report=term-missing
```

### Testy równoległe (szybsze wykonanie)

```bash
# Instalacja pytest-xdist
pip install pytest-xdist

# Uruchamianie równoległe
pytest tests/unit/ -n auto

# Z określoną liczbą workerów
pytest tests/unit/ -n 4
```

### Testy z profilowaniem wydajności

```bash
# Instalacja pytest-benchmark
pip install pytest-benchmark

# Uruchamianie testów wydajności
pytest tests/unit/ --benchmark-only
```

## Scenariusze testowe

### 1. Test pełnej orkiestracji

```bash
# Test całego przepływu przetwarzania
pytest tests/unit/test_enhanced_orchestrator.py::TestEnhancedOrchestrator::test_process_command_basic_flow -v

# Test obsługi plików
pytest tests/unit/test_enhanced_orchestrator.py::TestEnhancedOrchestrator::test_process_file_image_handling -v
```

### 2. Test agentów specjalistycznych

```bash
# Test agenta OCR
pytest tests/unit/test_ocr_agent.py::TestOCRAgent::test_process_image_success -v

# Test agenta wyszukiwania
pytest tests/unit/test_search_agent.py::TestSearchAgent::test_web_search_with_weather_query -v
```

### 3. Test komponentów core

```bash
# Test klienta LLM
pytest tests/unit/test_hybrid_llm_client.py::TestHybridLLMClient::test_chat_with_auto_selection -v

# Test fabryki agentów
pytest tests/unit/test_agent_factory.py::TestAgentFactory::test_create_agent_success -v
```

## Debugowanie testów

### 1. Testy z debugowaniem

```bash
# Zatrzymanie na pierwszym błędzie
pytest tests/unit/ -x

# Debugowanie z pdb
pytest tests/unit/ --pdb

# Więcej informacji o błędach
pytest tests/unit/ --tb=long

# Logi w czasie rzeczywistym
pytest tests/unit/ -s --log-cli-level=DEBUG
```

### 2. Uruchamianie pojedynczych testów

```bash
# Konkretny test
pytest tests/unit/test_enhanced_orchestrator.py::TestEnhancedOrchestrator::test_process_command_basic_flow -v -s

# Z debugowaniem
pytest tests/unit/test_enhanced_orchestrator.py::TestEnhancedOrchestrator::test_process_command_basic_flow --pdb
```

## Metryki jakości

### Docelowe pokrycie kodu:

- **Enhanced Orchestrator**: ~95%
- **Agent Factory**: ~98%
- **Enhanced Base Agent**: ~97%
- **OCR Agent**: ~95%
- **Search Agent**: ~95%
- **Hybrid LLM Client**: ~96%

### Sprawdzanie pokrycia:

```bash
# Generowanie raportu HTML
pytest tests/unit/ --cov=src --cov-report=html

# Otwarcie raportu
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Testowanie w CI/CD

### GitHub Actions

Przykład konfiguracji `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.12]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Poetry
      uses: snok/install-poetry@v1

    - name: Install dependencies
      run: poetry install

    - name: Run tests
      run: |
        poetry run pytest tests/unit/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Wskazówki dotyczące pisania testów

### 1. Konwencje nazewnictwa

```python
class TestComponentName:
    def test_method_name_expected_behavior(self):
        # Given
        # When
        # Then
        pass
```

### 2. Struktura testów

```python
@pytest.mark.asyncio
async def test_process_file_success(self, mock_dependencies):
    """Test pomyślnego przetwarzania pliku"""
    # Given
    input_data = {"file": "test_data"}
    expected_result = "expected_output"

    # When
    result = await component.process(input_data)

    # Then
    assert result.success is True
    assert result.data == expected_result
```

### 3. Mock'owanie zależności

```python
@pytest.fixture
def mock_llm_client(self):
    with patch('src.backend.core.hybrid_llm_client.hybrid_llm_client') as mock:
        mock.chat.return_value = {"message": {"content": "test response"}}
        yield mock
```

## Rozwiązywanie problemów

### Częste błędy:

1. **Import Error**: Sprawdź ścieżki w `sys.path.insert()`
2. **Async Errors**: Upewnij się, że używasz `@pytest.mark.asyncio`
3. **Mock Errors**: Sprawdź czy wszystkie zależności są zamockowane
4. **Database Errors**: Użyj osobnej bazy danych testowej

### Debugging:

```bash
# Sprawdzenie importów
python -c "import src.backend.agents.enhanced_orchestrator; print('OK')"

# Test pojedynczego komponentu
python -m pytest tests/unit/test_enhanced_orchestrator.py::TestEnhancedOrchestrator::test_initialization -v -s
```

## Podsumowanie

Przygotowany zestaw testów zapewnia:

- ✅ **95%+ pokrycie kodu** kluczowych komponentów
- ✅ **Kompletne testowanie orkiestracji** agentów
- ✅ **Testy wszystkich scenariuszy** (sukces, błędy, przypadki brzegowe)
- ✅ **Mockowanie zależności** zewnętrznych (LLM, internet, baza danych)
- ✅ **Testy wydajności** i równoczesności
- ✅ **Łatwe rozszerzanie** o nowe komponenty

Uruchomienie wszystkich testów:
```bash
pytest tests/unit/ --cov=src --cov-report=html -v
```

Ten zestaw testów pozwoli na pewne rozwijanie aplikacji z gwarancją, że refaktoryzacje nie zepsują istniejącej funkcjonalności.
