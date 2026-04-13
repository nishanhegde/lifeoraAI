from __future__ import annotations

import logging
from typing import List, Optional

from .embedder import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchResult:
    def __init__(self, text: str, metadata: dict, score: float):
        self.text = text
        self.metadata = metadata
        self.score = score  # cosine similarity in [0, 1]; higher = more relevant

    def __repr__(self) -> str:
        topic = self.metadata.get("topic", "?")
        return f"[{topic} | score={self.score:.3f}] {self.text[:80]}..."


class Retriever:
    """
    Query the vector store and return ranked chunks.

    Fix vs. original:
    - Uses vector_store.query() (public method) instead of
      vector_store._get_collection() (private — encapsulation breach).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        threshold: float = 0.3,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold

    def search(self, query: str, topic_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Embed the query and return top-K relevant chunks above the similarity threshold.
        Raises RetrievalError if the vector store query fails.
        """
        logger.debug("Searching for: %r (topic_filter=%s)", query[:80], topic_filter)

        query_embedding = self.embedder.embed_one(query)
        where = {"topic": topic_filter} if topic_filter else None

        # Uses the public query() method — no private access
        raw = self.vector_store.query(
            embedding=query_embedding,
            n_results=self.top_k,
            where=where,
        )

        results: List[SearchResult] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score in [0, 1]
            score = 1.0 - (dist / 2.0)
            if score >= self.threshold:
                results.append(SearchResult(text=doc, metadata=meta, score=score))

        logger.debug("Retrieved %d chunks (threshold=%.2f)", len(results), self.threshold)
        return results

    def format_context(self, results: List[SearchResult]) -> str:
        """Format retrieved chunks into a context block for the LLM prompt."""
        if not results:
            return "No relevant information found in the knowledge base."

        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{r.text}")

        return "\n\n".join(parts)
