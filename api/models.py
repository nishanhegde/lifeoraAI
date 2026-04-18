from typing import List, Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Health or lifestyle question")
    topic_filter: Optional[str] = Field(None, description="Restrict to: nutrition | exercise | lifestyle")
    show_sources: bool = Field(False, description="Include source chunk attribution in the response")


class IngestRequest(BaseModel):
    data_dir: Optional[str] = Field(None, description="Override data directory (defaults to config data.raw_dir)")


# ── Responses ─────────────────────────────────────────────────────────────────

class SourceItem(BaseModel):
    rank: int
    source: str
    score: float
    excerpt: str = Field(..., description="First 120 characters of the chunk")


class AskResponse(BaseModel):
    answer: str
    provider: str
    sources: Optional[List[SourceItem]] = None


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str                  # "ok" | "degraded"
    provider: str
    total_chunks: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
