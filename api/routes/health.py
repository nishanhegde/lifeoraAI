from fastapi import APIRouter
from api.dependencies import get_pipeline
from api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Liveness + readiness check.
    Returns provider name and total chunks in the vector store.
    Use this endpoint to verify the API is up before sending requests.
    """
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        provider=pipeline.provider.name,
        total_chunks=pipeline.vector_store.count(),
    )
