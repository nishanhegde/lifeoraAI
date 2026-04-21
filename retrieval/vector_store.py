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

    def ingest_jsonl(self, jsonl_path: str) -> int:
        """Embed and store chunks from a JSONL file exported by the health intel pipeline."""
        import json
        path = Path(jsonl_path)
        if not path.exists():
            raise IngestError(f"JSONL file not found: {jsonl_path}")

        # Use the JSONL's own chunk_id (UUID) as the ChromaDB ID — guaranteed unique per record.
        # Deduplicate by chunk_id in case the file itself has repeated entries.
        seen_ids: set = set()
        chunks = []
        ids = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not record.get("embedding_ready", True):
                    continue
                chunk_id = record["chunk_id"]
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                ids.append(chunk_id)
                chunks.append({
                    "text": record["text"],
                    "metadata": {
                        "source":          record.get("source_name", "unknown"),
                        "source_url":      record.get("source_url", ""),
                        "document_title":  record.get("document_title", ""),
                        "domain":          record.get("domain", ""),
                        "framework":       record.get("framework", ""),
                        "region":          record.get("region", ""),
                        "trust_level":     record.get("trust_level", ""),
                        "safety_category": record.get("safety_category", ""),
                        "audience":        record.get("audience", ""),
                        "topic":           record.get("topic", ""),
                    },
                })

        added = self._add_chunks_with_ids(chunks, ids)
        logger.info("JSONL ingest: %d records read → %d new chunks added", len(chunks), added)
        return added

    def _add_chunks_with_ids(self, chunks: List[dict], ids: List[str]) -> int:
        """Like add_chunks but uses caller-supplied IDs instead of computing content-hash IDs."""
        if not chunks:
            return 0
        try:
            collection = self._get_collection()
            existing = collection.get(ids=ids, include=[])
            existing_id_set = set(existing["ids"])

            new_pairs = [
                (cid, chunk)
                for cid, chunk in zip(ids, chunks)
                if cid not in existing_id_set
            ]
            if not new_pairs:
                return 0

            texts = [c["text"] for _, c in new_pairs]
            embeddings = self.embedder.embed(texts)

            for batch_start in range(0, len(new_pairs), _INSERT_BATCH_SIZE):
                batch = new_pairs[batch_start: batch_start + _INSERT_BATCH_SIZE]
                batch_embs = embeddings[batch_start: batch_start + _INSERT_BATCH_SIZE]
                collection.add(
                    ids=[cid for cid, _ in batch],
                    embeddings=batch_embs,
                    documents=[c["text"] for _, c in batch],
                    metadatas=[c["metadata"] for _, c in batch],
                )
            return len(new_pairs)
        except Exception as exc:
            raise IngestError(f"Failed to add chunks to vector store: {exc}") from exc

    def count(self) -> int:
        return self._get_collection().count()
