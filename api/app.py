import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core.exceptions import (
    ValidationError,
    ProviderError,
    ProviderUnavailableError,
    RetrievalError,
    IngestError,
    ConfigError,
)
from core.logging_config import setup_logging
from rag.pipeline import RAGPipeline
from api.dependencies import set_pipeline
from api.routes import ask, ingest, health

load_dotenv()
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise the RAG pipeline once at startup.
    If the LLM provider is unavailable (e.g. Ollama not running), the API
    still starts in degraded mode — the UI loads and /health reflects the
    issue. Requests to /ask return 503 until the provider is reachable.
    """
    logger.info("Starting LifeoraAI API...")
    try:
        pipeline = RAGPipeline(config_path="config.yaml")
    except ProviderUnavailableError as exc:
        logger.warning("Provider unavailable at startup: %s", exc)
        logger.warning("Starting in degraded mode — start Ollama then retry /ask requests.")
        pipeline = RAGPipeline(config_path="config.yaml", skip_provider_check=True)
    set_pipeline(pipeline)
    logger.info("RAG pipeline ready.")
    yield
    logger.info("LifeoraAI API shutting down.")


app = FastAPI(
    title="LifeoraAI API",
    description="Retrieval-Augmented Generation for health, nutrition, exercise, and lifestyle guidance.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins in dev. In production, replace "*" with your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers → HTTP status codes ────────────────────────────────────

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"error": "Validation failed", "detail": str(exc)})


@app.exception_handler(ProviderUnavailableError)
async def provider_unavailable_handler(request: Request, exc: ProviderUnavailableError):
    return JSONResponse(status_code=503, content={"error": "LLM provider unavailable", "detail": str(exc)})


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    return JSONResponse(status_code=503, content={"error": "LLM provider error", "detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError):
    return JSONResponse(status_code=500, content={"error": "Retrieval failed", "detail": str(exc)})


@app.exception_handler(IngestError)
async def ingest_error_handler(request: Request, exc: IngestError):
    return JSONResponse(status_code=500, content={"error": "Ingestion failed", "detail": str(exc)})


@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError):
    return JSONResponse(status_code=500, content={"error": "Configuration error", "detail": str(exc)})


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router, tags=["Health"])
app.include_router(ask.router, tags=["RAG"])
app.include_router(ingest.router, tags=["Ingest"])

# ── UI — serve from /ui, root redirects to chat ───────────────────────────────

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("ui/index.html")
