"""
Dependency injection for the RAGPipeline.
The pipeline is initialised once at startup (via lifespan) and injected
into every route handler — no re-initialisation per request.
"""
from rag.pipeline import RAGPipeline

_pipeline: RAGPipeline = None


def set_pipeline(pipeline: RAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> RAGPipeline:
    return _pipeline
