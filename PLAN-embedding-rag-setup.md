# LifeoraAI — Embedding Model & RAG Setup Plan

> **Goal:** Build a retrieval-driven answer system for health, food, exercise, and lifestyle guidance using trusted knowledge.
>
> **Approach:** Start fully local (Path A) — zero cost, full control, learn the fundamentals. Architecture stays provider-agnostic so switching to cloud APIs (Path B) is a config change, not a rewrite.

---

## Architecture: Provider-Agnostic Design

```
                    ┌─────────────────────────┐
  User Question ──> │   core/validation.py     │  length limits + injection detection
                    └──────────┬──────────────┘
                               │ validated query
                    ┌──────────▼──────────────┐
                    │   Retrieval Layer        │  same for ALL paths below
                    │   Embedder + ChromaDB    │  sentence-transformers + ChromaDB
                    └──────────┬──────────────┘
                               │ top-K chunks
                    ┌──────────▼──────────────┐
                    │   RAGPipeline            │  safe prompt assembly, structured logs
                    │   rag/pipeline.py        │
                    └──────────┬──────────────┘
                               │ prompt
                    ┌──────────▼──────────────┐
                    │   LLM Provider Interface │  abstract — swap without rewrite
                    │   rag/providers/base.py  │  thread-safe, retry, timeout on all impls
                    └──────────┬──────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         Path A (DONE)    Path B (DONE)   Path C (end goal)
         Ollama            Claude API      Lifeora SLM
         local, free       anthropic SDK   ┌─────────────────┐
         no GPU needed     retry on 429    │ Fine-tuned on:  │
                                           │ • health data   │
                                           │ • nutrition     │
                                           │ • exercise      │
                                           │ • lifestyle     │
                                           │                 │
                                           │ Baked-in:       │
                                           │ • Lifeora tone  │
                                           │ • safety rules  │
                                           │ • domain style  │
                                           │ • personalized  │
                                           │   responses     │
                                           └─────────────────┘
                                           Plugs into same
                                           RAG layer unchanged
```

**Key principle:** The LLM is behind an interface. Retrieval, knowledge, and orchestration code stays the same regardless of which LLM powers generation. Path C is the end goal — a Lifeora-specific SLM that replaces the generic model but reuses the entire RAG layer unchanged.

---

## Path C: Lifeora Fine-Tuned SLM (End Goal)

**What:** A small language model trained specifically for Lifeora's health, food, exercise, and lifestyle domain.

### What makes it "Lifeora SLM" vs a generic model

| Generic LLM (Path A/B) | Lifeora SLM (Path C) |
|------------------------|----------------------|
| Knows everything broadly | Deep health/lifestyle expertise |
| Generic tone | Lifeora voice and style |
| No safety guardrails | Built-in health safety rules |
| No personalization | Understands user goals, conditions |
| Fixed knowledge | RAG keeps it up-to-date |

### What it is trained on
- Curated health, nutrition, and exercise knowledge (knowledge base built in Phases 0–3)
- Good Q&A examples generated during Path A/B (RAG outputs reviewed and labelled)
- Safety examples: what to say, what to avoid, when to refer to a doctor
- Tone and format examples: how Lifeora should respond

### How it gets built (by the other team)
1. Start from a base model (Gemma 2 or Llama 3.1 — small, efficient)
2. Instruction tuning on domain Q&A examples
3. Preference tuning (RLHF/DPO) using user feedback from Path A/B
4. Safety fine-tuning on health-specific risk scenarios
5. Evaluated against Path B (Claude API) as quality benchmark

### How it connects back to RAG
- Path C SLM slots into the same `LLMProvider` interface (`rag/providers/base.py`)
- RAG layer is unchanged — same embeddings, same vector store, same retrieval
- Switch: `config.yaml: llm_provider: lifeora-slm`
- Result: fine-tuned domain knowledge (baked in) + live trusted knowledge (via RAG)

---

## Phase 0: Prerequisites & Local Setup — DONE ✅

### 0a: Environment
- [x] Python 3.11+ virtual environment
- [x] `pyproject.toml` — proper installable package; `pip install -e ".[dev]"` installs everything
- [x] No `sys.path` hacks — imports work correctly under gunicorn / uvicorn / Docker

### 0b: Local LLM via Ollama
- [x] Ollama installed and documented
- [x] `ollama serve` daemon setup
- [x] `timeout_seconds` configurable per provider in `config.yaml`

#### Model options (set `config.yaml: ollama.model`)

| Model | Size | RAM | Speed | Best For |
|-------|------|-----|-------|----------|
| `llama3.2:3b` | 2GB | 8GB | ⚡⚡⚡ | Recommended starting point |
| `gemma2:9b` | 5.5GB | 16GB | ⚡⚡ | Higher quality |
| `llama3.1:8b` | 4.7GB | 16GB | ⚡⚡ | Best local quality |
| `phi3.5` | 2.3GB | 8GB | ⚡⚡⚡ | Fast, instruction-tuned |

### 0c: Knowledge Base
- [x] `data/raw/nutrition.md` — protein, iron, healthy fats, carbs, vitamins, hydration
- [x] `data/raw/exercise.md` — strength training, cardio, muscle groups, recovery, hypertrophy
- [x] `data/raw/lifestyle.md` — sleep, stress management, gut health, weight, habit formation

### 0d: Project Structure
- [x] Final structure:
  ```
  lifeoraAI/
    core/           ← exceptions, structured logging, input validation
    data/raw/       ← source knowledge files (markdown)
    embeddings/     ← ChromaDB vector store (persisted to disk)
    retrieval/      ← embedder, vector store, search
    rag/pipeline.py ← end-to-end orchestration
    rag/providers/  ← OllamaProvider, ClaudeProvider, factory, base interface
    notebooks/      ← phase1_embeddings.ipynb
    tests/          ← retrieval, validation, encapsulation tests
    config.yaml     ← switch providers here
    pyproject.toml  ← installable package definition
  ```

---

## Phase 1: Embeddings — DONE ✅

**What:** Convert text into vectors so similar content can be found by meaning.

- [x] `retrieval/embedder.py` — `Embedder` wraps `sentence-transformers`
- [x] Model: `all-MiniLM-L6-v2` (384 dims, fast, no GPU needed)
- [x] Batched encoding — processes in groups of 64 to prevent OOM on large ingestions
- [x] `notebooks/phase1_embeddings.ipynb` — cosine similarity matrix, PCA 2D visualisation, live retrieval demo

**Key decisions made:**
- Embedding model: `all-MiniLM-L6-v2` — fast, good quality, no GPU
- Chunk size: 300 words with 50-word overlap (tunable in `config.yaml`)

---

## Phase 2: Vector Store — DONE ✅

**What:** Store embedded knowledge so it can be searched across sessions.

- [x] `retrieval/vector_store.py` — ChromaDB with cosine similarity, persists to `embeddings/chroma_db/`
- [x] `VectorStore.query()` — public method exposed; `_get_collection()` stays private (no encapsulation breach)
- [x] Batch inserts — 100 chunks per `collection.add()` call (not one at a time)
- [x] Content-hash IDs — `MD5(text)` as chunk ID; re-ingesting updated files replaces changed chunks automatically, no duplicates
- [x] Targeted dedup check — `collection.get(ids=candidates)` instead of fetching all IDs

---

## Phase 3: RAG Pipeline — DONE ✅

**What:** Wire retrieval to a local LLM so answers are grounded in your knowledge.

- [x] `rag/providers/base.py` — `LLMProvider` abstract interface + `GenerationConfig` (temperature, max_tokens, timeout)
- [x] `rag/providers/ollama_provider.py` — Path A: local Ollama backend
- [x] `rag/providers/factory.py` — builds provider from `config.yaml`, calls `is_available()` at startup (fail fast)
- [x] `rag/pipeline.py` — validates input → retrieves top-K chunks → assembles prompt → generates answer
- [x] Safe prompt assembly: `<<CONTEXT>>`/`<<QUESTION>>` replaced via `.replace()`, never `str.format()` with user input
- [x] `main.py` — `argparse` CLI: `ingest` | `ask` | `--sources` | `--debug` | `--config`

### Phase 3b: Production Hardening — DONE ✅

**What:** Make the system reliable and safe before real usage. All items below are complete.

**Reliability**
- [x] Thread safety — `threading.Lock` with double-checked locking on all provider client inits
- [x] Retry + backoff — `tenacity` retries transient errors (connection drop, Ollama restart, API rate limit) with exponential backoff
- [x] Timeouts — `timeout_seconds` on both providers; prevents hung requests blocking the process
- [x] Fail fast — `is_available()` at startup; missing API key or unpulled model surfaces immediately with a clear error

**Security**
- [x] Input validation — `core/validation.py`: min/max length, empty check, prompt injection pattern detection
- [x] Config validation — `validate_config()` at startup; bad `config.yaml` values raise `ConfigError` before any work begins
- [x] Safe templating — user input never passed to `str.format()`; custom delimiters prevent crashes on `{` or `}` in queries

**Observability**
- [x] Structured logging — `core/logging_config.py`: plain text for dev, JSON-lines for prod log aggregation
- [x] All `print()` removed — replaced with `logger.info/warning/debug` throughout
- [x] Named exceptions — `LifeoraError` hierarchy: `ValidationError`, `ProviderError`, `ProviderUnavailableError`, `RetrievalError`, `IngestError`, `ConfigError`

**Data quality**
- [x] Heading-aware chunking — `TextChunker` splits on markdown headings before word-chunking; section content stays coherent
- [x] Embedding batching — `Embedder.embed()` processes in batches of 64; no OOM on large ingestions

### Phase 3b: Claude Provider (Path B) — DONE ✅

- [x] `rag/providers/claude_provider.py` — same interface as Ollama, calls Anthropic SDK
- [x] Retries `RateLimitError`, `APIConnectionError`, `InternalServerError` with exponential backoff
- [x] Switch: `config.yaml: llm_provider: "claude"` — no code changes, no migration
- [x] `ANTHROPIC_API_KEY` loaded from `.env` via `python-dotenv`

---

## Phase 4: Smarter Retrieval — NEXT

**What:** Make search smarter — not just meaning, but also filters and ranking.

- [ ] Metadata filters: topic (nutrition / exercise / lifestyle), safety_category, audience level
- [ ] Hybrid search: semantic similarity + keyword matching (BM25)
- [ ] Re-ranking: score retrieved chunks by relevance before sending to LLM
- [ ] Handle user context (dietary restrictions, fitness level) in retrieval

---

## Phase 5: Domain Knowledge Graph — PLANNED

**What:** Model connections between entities for richer, context-aware retrieval.

- [ ] Map key relationships:
  - food → nutrients
  - exercise → muscle groups
  - goal → recommended habits
  - condition → cautions / exclusions
- [ ] Store as structured data (JSON or lightweight graph DB)
- [ ] Enrich retrieval: "building muscle" query also pulls protein-rich food data automatically

---

## Phase 6: Evaluation & Iteration — PLANNED

**What:** Measure if the system actually works well and iterate.

- [ ] Create a test set of 20–30 questions with expected good answers
- [ ] Evaluate: relevance of retrieved chunks, quality of generated answers
- [ ] Identify failure modes: wrong chunks retrieved, hallucinations, safety gaps
- [ ] Iterate on chunk size, embedding model, prompt template, retrieval strategy

---

## Tech Stack

| Component | Path A (done) | Path B (done) | Path C (future) |
|-----------|---------------|---------------|-----------------|
| Embeddings | `all-MiniLM-L6-v2` (local) | Same | Same |
| Vector Store | ChromaDB (local) | Same | Same |
| LLM | Ollama (local) | Claude API | Lifeora SLM |
| Retry | `tenacity` | `tenacity` | Same |
| Validation | `core/validation.py` | Same | Same |
| Logging | `core/logging_config.py` | Same | Same |
| Exceptions | `core/exceptions.py` | Same | Same |

### Hardware Requirements (Path A)
- **Minimum:** 8GB RAM, any modern CPU — runs `llama3.2:3b`, `phi3.5`
- **Recommended:** 16GB RAM — runs `gemma2:9b`, `llama3.1:8b` (best quality)
- **No GPU required** for inference via Ollama

---

## Notes

- **Production hardening (Phase 3b) is complete** — the system is thread-safe, retry-resilient, input-validated, and fully logged before Phase 4 work begins.
- **Data collection feeds Phase 4+** — richer metadata on knowledge chunks (topic, safety_category, audience) enables better filtering.
- **Path A → B is a config switch, not a migration.** The provider interface makes this seamless.
- **Path C SLM plugs in the same way.** The RAG layer does not change when the SLM is ready.
- Each phase is a conversation — build, test, validate, then move forward.
