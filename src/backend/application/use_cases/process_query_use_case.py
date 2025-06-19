from typing import Any

from src.backend.domain.repositories import FoodItemRepository, UserRepository
from src.backend.infrastructure.llm_api.llm_client import LLMClient
from src.backend.infrastructure.vector_store.vector_store_impl import (
    EnhancedVectorStoreImpl,
)


class ProcessQueryUseCase:
    def __init__(
        self,
        user_repository: UserRepository,
        food_item_repository: FoodItemRepository,
        llm_client: LLMClient,
        vector_store: EnhancedVectorStoreImpl,
    ):
        self.user_repository = user_repository
        self.food_item_repository = food_item_repository
        self.llm_client = llm_client
        self.vector_store = vector_store

    async def execute(self, query: str, user_id: str) -> Any:
        raise NotImplementedError("ProcessQueryUseCase.execute() not implemented")
