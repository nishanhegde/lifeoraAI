from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

# Embed at most this many texts per model.encode() call to avoid OOM
_EMBED_BATCH_SIZE = 64


class TextChunker:
    """
    Split markdown / plain-text documents into overlapping chunks.

    Improvements vs. original:
    - Respects markdown section boundaries (splits on headings) before
      word-chunking, so a heading and its body stay together.
    - Falls back to word-count chunking within each section.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size      # target words per chunk
        self.chunk_overlap = chunk_overlap

    # ── Public ───────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """Return list of {text, metadata} dicts."""
        metadata = metadata or {}
        sections = self._split_by_heading(text)
        chunks: List[dict] = []
        for section in sections:
            chunks.extend(self._word_chunk(section, metadata, start_index=len(chunks)))
        return chunks

    def chunk_file(self, file_path: Union[str, Path]) -> List[dict]:
        """Read a file and chunk it. Uses the filename stem as topic metadata."""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        topic = path.stem   # e.g. "nutrition", "exercise"
        return self.chunk_text(text, metadata={"source": path.name, "topic": topic})

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_by_heading(text: str) -> List[str]:
        """
        Split markdown on H1/H2/H3 headings so each section stays coherent.
        Falls back to the whole text if no headings are found.
        """
        parts = re.split(r"(?m)^#{1,3}\s+", text)
        # re.split drops the delimiter — prepend a space so content isn't lost
        return [p.strip() for p in parts if p.strip()]

    def _word_chunk(self, text: str, metadata: dict, start_index: int) -> List[dict]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**metadata, "chunk_index": start_index + len(chunks)},
                })
            if end == len(words):
                break
            start = end - self.chunk_overlap
        return chunks


class Embedder:
    """
    Wrap sentence-transformers to produce normalised vectors from text.

    Improvements vs. original:
    - Batched encoding (_EMBED_BATCH_SIZE) prevents OOM on large ingestions.
    - Lazy model load — model is only loaded when first needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of unit-normalised embedding vectors, batched to avoid OOM."""
        if not texts:
            return []
        model = self._get_model()
        all_embeddings: List[List[float]] = []
        for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[batch_start: batch_start + _EMBED_BATCH_SIZE]
            vecs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
