import logging
from fastapi import APIRouter
from api.dependencies import get_pipeline
from api.models import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest = None):
    """
    Chunk, embed, and store all documents from the knowledge base directory.
    Safe to call repeatedly — only new or changed content is added.

    - **data_dir**: optional override for the source directory (defaults to `config.yaml: data.raw_dir`)
    """
    pipeline = get_pipeline()
    logger.info("POST /ingest | data_dir=%s", request.data_dir if request else "default")

    added = pipeline.ingest(data_dir=request.data_dir if request else None)
    total = pipeline.vector_store.count()

    return IngestResponse(
        chunks_added=added,
        total_chunks=total,
        message=f"Ingestion complete. {added} new chunks added, {total} total in store.",
    )
