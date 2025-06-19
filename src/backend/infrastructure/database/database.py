from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.backend.config import settings

# Database URL is now loaded from central configuration.
# We're using SQLite which will store the database in a file.

# Create async SQLAlchemy engine
engine = create_async_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}, echo=False
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)

# Create base class for SQLAlchemy models
Base = declarative_base()


# Dependency to get an async database session
async def get_db() -> AsyncGenerator:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
