import json
import os
from typing import Any, List

import pytest

from src.backend.agents.prompts import get_entity_extraction_prompt
from src.backend.agents.tools import generate_clarification_question_text
from src.backend.agents.utils import extract_json_from_text
from src.backend.config import settings
from src.backend.core import crud
from src.backend.core.database import AsyncSessionLocal
from src.backend.core.llm_client import llm_client
from src.backend.models.shopping import Product, ShoppingTrip

# Ladowanie danych testowych bezposrednio z pliku JSON
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "test_data.json"
)
with open(TEST_DATA_PATH, "r") as f:
    TEST_DATA = json.load(f)


# --- NOWA FUNKCJA GENERUJĄCA PYTANIE ---
def generate_clarification_question(options: List[Any]) -> str:
    """
    Na podstawie listy potencjalnych obiektów, generuje pytanie do użytkownika.
    """
    if not options:
        return "Coś poszło nie tak, nie mam opcji do wyboru."

    # Krok 1: Stwórz czytelne opisy opcji
    formatted_options = []
    for i, obj in enumerate(options, 1):
        if isinstance(obj, ShoppingTrip):
            formatted_options.append(
                f"{i}. Paragon ze sklepu '{obj.store_name}' " f"z dnia {obj.trip_date}."
            )
        elif isinstance(obj, Product):
            formatted_options.append(
                f"{i}. Produkt '{obj.name}' w cenie {obj.unit_price} zł."
            )

    options_text = "\n".join(formatted_options)

    # Krok 2: Na razie, dla testów, zwracamy prosty, sformatowany tekst.
    # W przyszłości ten tekst można przekazać do LLM, aby ubrał go w ładniejsze zdanie.
    return f"ZNALEZIONO KILKA OPCJI. Proszę, wybierz jedną:\n{options_text}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent, user_prompt",
    [(item["intent"], item["prompt"]) for item in TEST_DATA],
)
async def test_entity_extraction_parametrized(intent: str, user_prompt: str) -> None:
    """
    Testuje pełny przepływ: ekstrakcję, wyszukiwanie i podejmowanie decyzji,
    w tym generowanie pytania doprecyzowującego.
    """
    print(f"\n--- Testuję polecenie: '{user_prompt}' (Intencja: {intent}) ---")

    try:
        # Krok 1: Ekstrakcja encji z LLM
        prompt = get_entity_extraction_prompt(user_prompt, intent)
        messages = [
            {
                "role": "system",
                "content": "Jesteś precyzyjnym systemem ekstrakcji encji. Zawsze zwracaj tylko JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await llm_client.chat(
            model=settings.DEFAULT_CHAT_MODEL,
            messages=messages,
            stream=False,
            options={"temperature": 0.0},
        )

        raw_response = response["message"]["content"]
        parsed_json = extract_json_from_text(raw_response)

        assert parsed_json is not None, "Nie znaleziono JSON w odpowiedzi"
        print("Krok 1: Ekstrakcja danych z LLM zakończona sukcesem.")
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))

        # Krok 2: Wyszukiwanie w bazie i logika decyzyjna
        print("\nKrok 2: Wyszukiwanie rekordów w bazie danych...")
        async with AsyncSessionLocal() as db:
            znalezione_obiekty = []
            if intent == "CZYTAJ_PODSUMOWANIE":
                pass
            elif intent == "DODAJ_ZAKUPY":
                pass
            elif intent in ["UPDATE_ITEM", "DELETE_ITEM", "READ_ITEM"]:
                znalezione_obiekty = await crud.find_item_for_action(
                    db, entities=parsed_json
                )
            elif intent in ["UPDATE_PURCHASE", "DELETE_PURCHASE", "READ_PURCHASE"]:
                znalezione_obiekty = await crud.find_purchase_for_action(
                    db, entities=parsed_json
                )

            if intent not in ["CZYTAJ_PODSUMOWANIE", "DODAJ_ZAKUPY"]:
                if len(znalezione_obiekty) > 1:
                    pytanie = generate_clarification_question_text(znalezione_obiekty)
                    assert pytanie is not None

    except Exception as e:
        pytest.fail(f"Wystąpił krytyczny błąd: {e}")
