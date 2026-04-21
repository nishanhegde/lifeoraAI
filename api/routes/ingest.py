import logging
from fastapi import APIRouter
from api.dependencies import get_pipeline
from api.models import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_JSONL = "data/processed/chunks_export.jsonl"


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest = None):
    """
    Embed and store health knowledge chunks from the JSONL corpus.
    Safe to call repeatedly — only new chunks are added (deduplication by content hash).

    - **jsonl_path**: optional override (defaults to `data/processed/chunks_export.jsonl`)
    """
    pipeline = get_pipeline()
    jsonl_path = (request.jsonl_path if request and request.jsonl_path else None) or _DEFAULT_JSONL
    logger.info("POST /ingest | jsonl_path=%s", jsonl_path)

    added = pipeline.ingest_jsonl(jsonl_path)
    total = pipeline.vector_store.count()

    return IngestResponse(
        chunks_added=added,
        total_chunks=total,
        message=f"Ingestion complete. {added} new chunks added, {total} total in store.",
    )
