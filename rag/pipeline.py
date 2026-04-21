from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from core.validation import validate_query, validate_topic_filter, validate_config
from retrieval import Embedder, TextChunker, VectorStore, Retriever
from rag.providers import get_provider, LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are LifeoraAI, a trusted health and wellness assistant.
You answer questions about nutrition, exercise, sleep, and lifestyle using only the provided context.
Be concise, accurate, and friendly. If the context does not contain enough information to answer,
say so clearly — never guess or hallucinate facts.
For any medical conditions or symptoms always recommend consulting a qualified doctor."""

# Use a plain string template with non-Python-format delimiters.
# This is intentional: user input may contain { } characters which would
# break str.format(), and we never want the user controlling substitution.
_USER_PROMPT_TEMPLATE = (
    "Context from LifeoraAI knowledge base:\n"
    "<<CONTEXT>>\n\n"
    "User question: <<QUESTION>>\n\n"
    "Answer based on the context above:"
)


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    1. Validate and clean user input
    2. Embed question → retrieve relevant chunks from ChromaDB
    3. Build prompt with safe string substitution
    4. Generate grounded answer via the configured LLM provider
    """

    def __init__(self, config_path: str = "config.yaml", skip_provider_check: bool = False):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at: {config_path.resolve()}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        validate_config(self.config)   # fail fast on bad config

        self._base_dir = config_path.parent

        emb_cfg = self.config.get("embedding", {})
        ret_cfg = self.config.get("retrieval", {})
        vs_cfg = self.config.get("vector_store", {})

        self.embedder = Embedder(model_name=emb_cfg.get("model", "all-MiniLM-L6-v2"))
        self.chunker = TextChunker(
            chunk_size=emb_cfg.get("chunk_size", 300),
            chunk_overlap=emb_cfg.get("chunk_overlap", 50),
        )

        persist_dir = self._base_dir / vs_cfg.get("persist_dir", "embeddings/chroma_db")
        self.vector_store = VectorStore(
            persist_dir=str(persist_dir),
            collection_name=vs_cfg.get("collection_name", "lifeoraai_knowledge"),
            embedder=self.embedder,
        )

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=ret_cfg.get("top_k", 5),
            threshold=ret_cfg.get("similarity_threshold", 0.3),
        )

        self.provider: LLMProvider = get_provider(self.config, check_available=not skip_provider_check)

        chunk_count = self.vector_store.count()
        logger.info(
            "LifeoraAI RAG ready | provider=%s | chunks=%d",
            self.provider.name,
            chunk_count,
        )
        if chunk_count == 0:
            logger.warning("Vector store is empty. Run: python main.py ingest")

    def ingest(self, data_dir: str = None) -> int:
        """Load and embed all documents in data_dir."""
        if data_dir is None:
            data_dir = self._base_dir / self.config.get("data", {}).get("raw_dir", "data/raw")
        logger.info("Ingesting documents from %s", data_dir)
        total = self.vector_store.ingest_directory(str(data_dir), self.chunker)
        logger.info("Ingestion complete. New chunks: %d | Total: %d", total, self.vector_store.count())
        return total

    def ingest_jsonl(self, jsonl_path: str) -> int:
        """Load and embed chunks from a JSONL file."""
        logger.info("Ingesting JSONL from %s", jsonl_path)
        total = self.vector_store.ingest_jsonl(jsonl_path)
        logger.info("JSONL ingestion complete. New chunks: %d | Total: %d", total, self.vector_store.count())
        return total

    def ask(
        self,
        question: str,
        topic_filter: Optional[str] = None,
        show_sources: bool = False,
    ) -> str:
        """
        Ask a health/lifestyle question, get a grounded answer.

        Args:
            question:     The user's question in plain English.
            topic_filter: Restrict retrieval to a topic ("nutrition", "exercise", "lifestyle").
            show_sources: If True, append source attribution to the answer.

        Raises:
            ValidationError:  Input failed validation.
            ProviderError:    LLM call failed after retries.
            RetrievalError:   Vector store query failed.
        """
        question = validate_query(question)
        topic_filter = validate_topic_filter(topic_filter)

        logger.info("Query received | topic_filter=%s | length=%d", topic_filter, len(question))

        results = self.retriever.search(question, topic_filter=topic_filter)
        context = self.retriever.format_context(results)

        # Safe substitution — never use str.format() with user-controlled input
        user_prompt = (
            _USER_PROMPT_TEMPLATE
            .replace("<<CONTEXT>>", context)
            .replace("<<QUESTION>>", question)
        )

        answer = self.provider.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        logger.info("Answer generated | provider=%s | sources=%d", self.provider.name, len(results))

        if show_sources and results:
            source_lines = "\n".join(
                f"  [{i}] {r.metadata.get('source', 'unknown')} (score={r.score:.2f})"
                for i, r in enumerate(results, 1)
            )
            answer += f"\n\nSources:\n{source_lines}"

        return answer
