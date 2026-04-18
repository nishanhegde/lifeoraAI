import logging
from fastapi import APIRouter
from api.dependencies import get_pipeline
from api.models import AskRequest, AskResponse, SourceItem

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a health or lifestyle question. Returns a grounded answer from the knowledge base.

    - **question**: plain-English question (3–2000 characters)
    - **topic_filter**: optional — restrict retrieval to `nutrition`, `exercise`, or `lifestyle`
    - **show_sources**: optional — include the source chunks used to generate the answer
    """
    pipeline = get_pipeline()
    logger.info("POST /ask | topic=%s | q=%r", request.topic_filter, request.question[:80])

    results = pipeline.retriever.search(request.question, topic_filter=request.topic_filter)
    context = pipeline.retriever.format_context(results)

    from rag.pipeline import SYSTEM_PROMPT, _USER_PROMPT_TEMPLATE
    user_prompt = (
        _USER_PROMPT_TEMPLATE
        .replace("<<CONTEXT>>", context)
        .replace("<<QUESTION>>", request.question)
    )
    answer = pipeline.provider.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    sources = None
    if request.show_sources:
        sources = [
            SourceItem(
                rank=i,
                source=r.metadata.get("source", "unknown"),
                score=round(r.score, 3),
                excerpt=r.text[:120],
            )
            for i, r in enumerate(results, 1)
        ]

    return AskResponse(answer=answer, provider=pipeline.provider.name, sources=sources)
