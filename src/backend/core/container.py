from dependency_injector import containers, providers

from src.backend.application.use_cases.process_query_use_case import ProcessQueryUseCase
from src.backend.infrastructure.database.database import get_db
from src.backend.infrastructure.database.repositories_impl import (
    SQLAlchemyFoodItemRepository,
    SQLAlchemyUserRepository,
)
from src.backend.infrastructure.llm_api.llm_client import LLMClient
from src.backend.infrastructure.vector_store.vector_store_impl import (
    EnhancedVectorStoreImpl,
)


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    db_session_provider = providers.Callable(get_db)

    user_repo_impl = providers.Factory(
        SQLAlchemyUserRepository, session=db_session_provider
    )
    food_item_repo_impl = providers.Factory(
        SQLAlchemyFoodItemRepository, session=db_session_provider
    )
    llm_client = providers.Singleton(LLMClient, api_key=config.llm_api_key)
    vector_store = providers.Singleton(EnhancedVectorStoreImpl, llm_client=llm_client)
    process_query_use_case = providers.Factory(
        ProcessQueryUseCase,
        user_repository=user_repo_impl,
        food_item_repository=food_item_repo_impl,
        llm_client=llm_client,
        vector_store=vector_store,
    )
