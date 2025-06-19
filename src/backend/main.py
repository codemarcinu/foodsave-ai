import logging
import os
import sys
from contextlib import asynccontextmanager

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.sql import text
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.backend.api import agents, chat, food, pantry, upload
from src.backend.api.v1.endpoints import receipts
from src.backend.api.v2.endpoints import receipts as receipts_v2
from src.backend.api.v2.exceptions import APIErrorDetail, APIException
from src.backend.application.use_cases.process_query_use_case import ProcessQueryUseCase
from src.backend.config import settings
from src.backend.core.container import Container
from src.backend.core.exceptions import ErrorCodes, ErrorDetail, FoodSaveException
from src.backend.core.migrations import run_migrations
from src.backend.core.seed_data import seed_database
from src.backend.infrastructure.database.database import AsyncSessionLocal, Base, engine

# Dodaj katalog projektu do PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Rate limiting ---
limiter = Limiter(key_func=get_remote_address)


# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)


# --- Middleware: Structured Logging ---
class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Clear any existing context
        clear_contextvars()

        # Bind request context
        bind_contextvars(
            request_id=request.headers.get("X-Request-ID", "none"),
            method=request.method,
            path=request.url.path,
            query=dict(request.query_params),
            client_ip=request.client.host if request.client else None,
        )

        logger = structlog.get_logger()
        logger.info("request.start")

        try:
            response = await call_next(request)

            # Log response
            bind_contextvars(
                status_code=response.status_code,
                response_size=response.headers.get("content-length", 0),
            )
            logger.info("request.complete")

            return response
        except Exception as e:
            logger.error("request.error", error=str(e))
            raise
        finally:
            clear_contextvars()


# --- Middleware: Auth (szkielet) ---
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Tu można dodać logikę JWT lub innej autoryzacji
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables asynchronously
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Run migrations asynchronously
    await run_migrations()

    # Seed the database asynchronously
    logger = structlog.get_logger()
    logger.info("database.seeding.start")
    async with AsyncSessionLocal() as db:
        try:
            await seed_database(db)
        except Exception as e:
            logger.error("database.seeding.error", error=str(e))
            raise
    logger.info("database.seeding.complete")
    yield  # Cleanup code here if needed


# Initialize DI container
container = Container()
container.config.from_dict(
    {"llm_api_key": settings.LLM_API_KEY, "database_url": settings.DATABASE_URL}
)

app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    description="Backend dla modułowej aplikacji agentowej.",
    version=settings.APP_VERSION,
)

# Attach container to app
app.container = container

# Wire dependencies
container.wire(modules=[__name__])

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    # UWAGA: W środowisku produkcyjnym te wartości powinny pochodzić ze zmiennych środowiskowych!
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StructuredLoggingMiddleware)
# app.add_middleware(AuthMiddleware)  # Odkomentuj, gdy AuthMiddleware będzie gotowe
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


# --- Exception Handlers ---
@app.exception_handler(FoodSaveException)
async def foodsave_exception_handler(request: Request, exc: FoodSaveException):
    """Handle FoodSave exceptions with standardized format"""
    logging.error(f"FoodSave error: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


# Add handler for API v2 exceptions
@app.exception_handler(APIException)
async def api_v2_exception_handler(request: Request, exc: APIException):
    """Handle API v2 exceptions with standardized format"""
    logging.error(f"API v2 error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorDetail(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        ).dict(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Fallback handler for all other exceptions"""
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            code=ErrorCodes.INTERNAL_ERROR,
            message="Internal Server Error",
            details={"exception": str(exc)},
        ).dict(),
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Standard 404 handler"""
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(
            code=ErrorCodes.NOT_FOUND,
            message="Not Found",
            details={"path": request.url.path},
        ).dict(),
    )


# --- API Versioning ---
api_v1 = APIRouter()
api_v1.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_v1.include_router(agents.router, tags=["Agents"])
api_v1.include_router(food.router)
api_v1.include_router(upload.router, tags=["Upload"])
api_v1.include_router(pantry.router, tags=["Pantry"])
api_v1.include_router(receipts.router)

api_v2 = APIRouter()
api_v2.include_router(receipts_v2.router)

app.include_router(api_v1, prefix="/api/v1")
app.include_router(api_v2, prefix="/api/v2")


# --- Health check ---
@app.get("/health", tags=["Health"])
@inject
async def health_check(
    process_query_uc: ProcessQueryUseCase = Depends(
        Provide[Container.process_query_use_case]
    ),
):
    """Health check endpoint with dependency injection example."""
    # Example of using injected dependency
    test_result = await process_query_uc.execute("health check", "system")
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "test_query_result": test_result,
    }


@app.get("/ready", tags=["Health"])
async def ready_check():
    """Readiness check (np. sprawdzenie połączenia z bazą)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "not ready", "error": str(e)}
        )


# --- Background task example ---
@app.post("/api/v1/long-task")
async def long_task(background_tasks: BackgroundTasks):
    """Przykład endpointu z background taskiem."""

    def do_long_work():
        import time

        time.sleep(5)
        logging.info("Long task finished!")

    background_tasks.add_task(do_long_work)
    return {"status": "started"}


@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint."""
    return {"message": f"Witaj w backendzie aplikacji {settings.APP_NAME}!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
