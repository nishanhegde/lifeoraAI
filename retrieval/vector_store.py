from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from core.exceptions import IngestError, RetrievalError
from .embedder import Embedder, TextChunker

logger = logging.getLogger(__name__)

# Max chunks sent to ChromaDB in a single add() call
_INSERT_BATCH_SIZE = 100


class VectorStore:
    """
    ChromaDB-backed vector store.

    Fixes vs. original:
    - query() method exposed publicly; _get_collection() stays private
    - Batch inserts (_INSERT_BATCH_SIZE) instead of one-at-a-time
    - Content-hash IDs: updating a file replaces old chunks automatically
    - Dedup check targets only candidate IDs, not the full collection
    - Errors surfaced as IngestError / RetrievalError
    """

    def __init__(self, persist_dir: str, collection_name: str, embedder: Embedder):
        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name
        self.embedder = embedder
        self._collection = None

    # ── Private ──────────────────────────────────────────────────────────────

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @staticmethod
    def _chunk_id(chunk: dict) -> str:
        """
        Stable content-hash ID.
        Same text always gets the same ID → re-ingesting updated files
        automatically replaces changed chunks without duplication.
        """
        source = chunk["metadata"].get("source", "doc")
        content_hash = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()[:12]
        return f"{source}_{content_hash}"

    # ── Public ───────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[dict]) -> int:
        """
        Embed and store chunks. Returns the count of newly added chunks.
        Skips chunks whose content-hash ID already exists.
        """
        if not chunks:
            return 0

        try:
            collection = self._get_collection()

            # Compute candidate IDs first (no embedding yet)
            candidate_ids = [self._chunk_id(c) for c in chunks]

            # Check only the candidate IDs — not the entire collection
            existing = collection.get(ids=candidate_ids, include=[])
            existing_id_set = set(existing["ids"])

            new_chunks = [
                (chunk_id, chunk)
                for chunk_id, chunk in zip(candidate_ids, chunks)
                if chunk_id not in existing_id_set
            ]

            if not new_chunks:
                return 0

            # Embed only the new chunks
            texts = [c["text"] for _, c in new_chunks]
            embeddings = self.embedder.embed(texts)

            # Batch insert
            for batch_start in range(0, len(new_chunks), _INSERT_BATCH_SIZE):
                batch = new_chunks[batch_start: batch_start + _INSERT_BATCH_SIZE]
                batch_embs = embeddings[batch_start: batch_start + _INSERT_BATCH_SIZE]
                collection.add(
                    ids=[cid for cid, _ in batch],
                    embeddings=batch_embs,
                    documents=[c["text"] for _, c in batch],
                    metadatas=[c["metadata"] for _, c in batch],
                )

            return len(new_chunks)

        except Exception as exc:
            raise IngestError(f"Failed to add chunks to vector store: {exc}") from exc

    def query(
        self,
        embedding: List[float],
        n_results: int,
        where: dict = None,
    ) -> dict:
        """
        Run a nearest-neighbour query against the collection.
        Returns raw ChromaDB result dict.
        Raises RetrievalError on failure.
        """
        try:
            collection = self._get_collection()
            return collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(f"Vector store query failed: {exc}") from exc

    def ingest_directory(self, data_dir: str, chunker: TextChunker = None) -> int:
        """Chunk and embed all .md and .txt files in data_dir."""
        chunker = chunker or TextChunker()
        total = 0
        data_path = Path(data_dir)

        if not data_path.exists():
            raise IngestError(f"Data directory does not exist: {data_dir}")

        for path in sorted(data_path.glob("**/*.md")) + sorted(data_path.glob("**/*.txt")):
            try:
                chunks = chunker.chunk_file(path)
                added = self.add_chunks(chunks)
                logger.info("Ingested %s: %d chunks, %d new", path.name, len(chunks), added)
                total += added
            except Exception as exc:
                raise IngestError(f"Failed to ingest {path.name}: {exc}") from exc

        return total

    def count(self) -> int:
        return self._get_collection().count()
