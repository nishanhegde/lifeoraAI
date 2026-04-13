# LifeoraAI

Production-grade Retrieval-Augmented Generation (RAG) system for health, nutrition, exercise, and lifestyle guidance.

---

## Overview

LifeoraAI answers questions about wellness by grounding every response in a curated knowledge base — not LLM hallucination. The architecture is provider-agnostic: swap between a free local model and the Claude API by changing one line in `config.yaml`, with no code changes.

Built for production: thread-safe, retry-resilient, input-validated, fully logged, and structured around named exceptions.

```
User Question
     │
     ▼
Input validation + injection detection (core/validation.py)
     │
     ▼
Embed question — sentence-transformers, batched (retrieval/embedder.py)
     │
     ▼
Nearest-neighbour search — ChromaDB cosine similarity (retrieval/vector_store.py)
     │
     ▼
Top-K chunks above similarity threshold (retrieval/search.py)
     │
     ▼
Safe prompt assembly — no str.format() with user input (rag/pipeline.py)
     │
     ▼
LLM Provider — thread-safe, retry + timeout (rag/providers/)
     │
     ▼
Grounded answer returned
```

---

## Roadmap

| Path | LLM | Status |
|------|-----|--------|
| A | Ollama (local, free) | Active — start here |
| B | Claude API (cloud) | Ready — one config switch |
| C | Lifeora fine-tuned SLM | Future — plugs into same RAG layer unchanged |

---

## Project Structure

```
lifeoraAI/
├── pyproject.toml           # installable package — fixes imports in all environments
├── config.yaml              # switch providers, tune retrieval settings
├── requirements.txt
├── main.py                  # argparse CLI: ingest | ask | --debug | --sources
├── .env.example             # copy to .env for API keys
│
├── core/                    # shared infrastructure
│   ├── exceptions.py        # LifeoraError, ValidationError, ProviderError, RetrievalError …
│   ├── logging_config.py    # structured logging — plain text (dev) or JSON-lines (prod)
│   └── validation.py        # input length limits, prompt injection detection, config validation
│
├── data/raw/                # source knowledge files (markdown)
│   ├── nutrition.md
│   ├── exercise.md
│   └── lifestyle.md
│
├── retrieval/
│   ├── embedder.py          # TextChunker (heading-aware) + Embedder (batched, lazy-load)
│   ├── vector_store.py      # ChromaDB — batch inserts, content-hash dedup, public query()
│   └── search.py            # Retriever — uses store.query(), never _get_collection()
│
├── rag/
│   ├── pipeline.py          # RAGPipeline — validates input, safe templating, structured logs
│   └── providers/
│       ├── base.py          # LLMProvider abstract interface + GenerationConfig
│       ├── ollama_provider.py   # thread-safe, retry on connection errors, configurable timeout
│       ├── claude_provider.py  # thread-safe, retry on rate limits, configurable timeout
│       └── factory.py       # builds provider from config.yaml, fails fast via is_available()
│
├── notebooks/
│   └── phase1_embeddings.ipynb  # embeddings walkthrough: similarity matrix, PCA, live demo
│
└── tests/
    └── test_retrieval.py    # chunker, embedder, vector store, retriever, validation tests
```

---

## Quick Start

### 1. Set up the environment

```bash
cd lifeoraAI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # installs all deps including pytest + jupyter
```

### 2. Set up a local LLM (Path A — free, no API key)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2:3b          # ~2GB download, runs on 8GB RAM
ollama serve                     # keep running in a separate terminal
```

Other model options — set `ollama.model` in `config.yaml`:

| Model | Size | RAM | Notes |
|-------|------|-----|-------|
| `llama3.2:3b` | 2GB | 8GB | Recommended starting point |
| `gemma2:9b` | 5.5GB | 16GB | Higher quality |
| `llama3.1:8b` | 4.7GB | 16GB | Best local quality |
| `phi3.5` | 2.3GB | 8GB | Fast, instruction-tuned |

### 3. Ingest the knowledge base

```bash
python main.py ingest
```

Chunks, embeds, and stores all files in `data/raw/` into ChromaDB. Run once — the store persists to `embeddings/chroma_db/`. Re-running is safe: only new or changed content is added (content-hash deduplication).

### 4. Ask questions

```bash
python main.py ask               # interactive Q&A loop
python main.py ask --sources     # show which chunks were used per answer
python main.py ask --debug       # verbose logging for troubleshooting
```

Example session:
```
You: What foods are high in iron?
LifeoraAI: Heme iron sources include red meat, liver, chicken, and fish.
Non-heme iron (plant-based) is found in lentils, spinach, tofu, and
fortified cereals. Pair with vitamin C to significantly improve absorption...

You: How much protein do I need to build muscle?
LifeoraAI: For muscle building, aim for 1.6–2.2g of protein per kilogram
of body weight per day. Good sources include chicken breast, eggs, Greek
yogurt, lentils, and tofu...
```

---

## Switching to Claude API (Path B)

1. Copy `.env.example` to `.env` and fill in your key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Change one line in `config.yaml`:
   ```yaml
   llm_provider: "claude"   # was "ollama"
   ```

3. Run normally — no code changes needed:
   ```bash
   python main.py ask
   ```

The retrieval layer (embeddings, ChromaDB) is completely unchanged. Only the generation step switches providers.

---

## Configuration

All settings live in `config.yaml`:

```yaml
llm_provider: "ollama"          # "ollama" | "claude"

ollama:
  model: "llama3.2:3b"
  temperature: 0.2
  max_tokens: 1024
  timeout_seconds: 60           # max wait per generation call

claude:
  model: "claude-sonnet-4-6"
  temperature: 0.2
  max_tokens: 1024
  timeout_seconds: 60

embedding:
  model: "all-MiniLM-L6-v2"    # 384-dim, fast, no GPU needed
  chunk_size: 300               # words per chunk
  chunk_overlap: 50             # words shared between consecutive chunks

retrieval:
  top_k: 5                      # chunks retrieved per query
  similarity_threshold: 0.3     # minimum cosine similarity to include a chunk
```

Config is validated at startup — bad values raise `ConfigError` immediately rather than failing silently on the first request.

---

## Production Design Decisions

### Thread safety
Both `OllamaProvider` and `ClaudeProvider` use double-checked locking (`threading.Lock`) on client initialisation. Safe under concurrent requests from a web server.

### Retry & timeout
Transient failures (connection drops, rate limits) are retried automatically with exponential backoff via `tenacity`. Every provider call has a configurable `timeout_seconds` to prevent hung requests.

### Input validation
All user queries pass through `core/validation.py` before any embedding or LLM call:
- Minimum/maximum length enforced
- Prompt injection patterns detected and rejected
- Topic filter values checked against an allowlist

### Safe prompt assembly
The LLM prompt uses `<<CONTEXT>>` / `<<QUESTION>>` delimiters replaced via `.replace()` — never `str.format()` with user input, which crashes on `{` or `}` characters.

### Content-hash deduplication
Chunk IDs are `MD5(content)` — if a source file is updated, changed chunks get new IDs and are added; unchanged chunks are skipped. No manual re-indexing needed.

### Fail fast
`is_available()` is called on every provider at startup. Missing API key or model not pulled surfaces immediately with a clear error, not on the first user request.

### Structured logging
`core/logging_config.py` provides plain-text format for local development and JSON-lines format for production log aggregation. All `print()` statements replaced with `logger.info/warning/debug`.

### Named exceptions
All errors are typed subclasses of `LifeoraError` (`ValidationError`, `ProviderError`, `RetrievalError`, `IngestError`, `ConfigError`). API layers can catch exactly what they handle.

---

## Adding Knowledge

Drop `.md` or `.txt` files into `data/raw/` then re-run ingest:

```bash
python main.py ingest
```

Only new or changed chunks are added — safe to run repeatedly.

---

## Running Tests

```bash
pytest tests/ -v
```

Covers: heading-aware chunking, embedding shape and normalisation, similarity ordering, large-batch embedding, vector store count and deduplication, topic filtering, retriever ranking, threshold filtering, encapsulation (Retriever must not call `_get_collection()` directly), input validation, and prompt injection detection.

---

## Phase 1 Notebook

```bash
jupyter notebook notebooks/phase1_embeddings.ipynb
```

Covers: text → vector conversion, cosine similarity matrix, PCA 2D visualisation, and a live retrieval demo against sample health chunks.

---

## Development Phases

| Phase | What | Status |
|-------|------|--------|
| 0 | Environment, Ollama, test data, project structure | Done |
| 1 | Embeddings understanding (notebook) | Done |
| 2 | ChromaDB vector store | Done |
| 3 | End-to-end RAG pipeline + Claude provider | Done |
| 3b | Production hardening — thread safety, retry, validation, logging | Done |
| 4 | Metadata filtering, hybrid search, re-ranking | Next |
| 5 | Domain knowledge graph (food → nutrients, goal → habits) | Planned |
| 6 | Evaluation set + iteration | Planned |
| C | Lifeora fine-tuned SLM — slots into same RAG layer | Future |
