from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Główna klasa do zarządzania ustawieniami aplikacji.
    Ustawienia są wczytywane ze zmiennych środowiskowych lub pliku .env.
    """

    APP_NAME: str = "Osobisty Asystent AI"
    APP_VERSION: str = "0.1.0"

    # Environment configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Konfiguracja dla klienta Ollama
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:12b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_CODE_MODEL: str = (
        "SpeakLeash/bielik-11b-v2.3-instruct:Q6_K"  # Polski model do zadań specjalnych
    )
    DEFAULT_CHAT_MODEL: str = "gemma3:12b"  # Model do ogólnej konwersacji
    DEFAULT_EMBEDDING_MODEL: str = "nomic-embed-text:latest"

    # Konfiguracja bazy danych
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/foodsave.db"

    # API key for LLM service
    LLM_API_KEY: str = ""

    # Konfiguracja Tesseract OCR
    TESSDATA_PREFIX: str = "/usr/share/tesseract-ocr/5/"

    # Ta linia mówi Pydantic, aby wczytał zmienne z pliku .env w głównym katalogu
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Tworzymy jedną, globalną instancję ustawień,
# której będziemy używać w całej aplikacji.
settings = Settings()
