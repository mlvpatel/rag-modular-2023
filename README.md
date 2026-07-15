# rag-modular-2023

Modular production RAG: hybrid retrieval fused in one SQL query, streaming answers, conversational memory, and a measured retrieval gate.

Part of the RAG line, a series of reference enterprise RAG implementations, one per retrieval strategy. This repository is the Modular (2023) rung. See [the line](#the-rag-line) below.

This service answers questions about your own documents. It runs dense vector search and sparse keyword search inside Postgres, fuses them with Reciprocal Rank Fusion in a single query, reranks with a cross encoder, streams grounded answers token by token, remembers the conversation, and runs fully locally at no cost or against cloud models in production.

![Answering a document question live on a local model](assets/videos/ragflowpro-demo.gif)

The animation above is a live, unedited run. The model is a local llama3.2, the documents (including a real SEC 10-K) are indexed in pgvector, and the answer streams in grounded in them. No paid keys were used.

Full recording at [assets/videos/ragflowpro-demo.webm](assets/videos/ragflowpro-demo.webm), full resolution screenshot at [assets/screenshots/ragflowpro-sample-data-demo.png](assets/screenshots/ragflowpro-sample-data-demo.png).

[![CI](https://github.com/mlvpatel/rag-modular-2023/actions/workflows/ci.yml/badge.svg)](https://github.com/mlvpatel/rag-modular-2023/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) ![Postgres](https://img.shields.io/badge/Postgres-pgvector-blue) ![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

## What this rung adds

The rung below, [rag-advanced-2023](https://github.com/mlvpatel/rag-advanced-2023), already does hybrid retrieval and reranking, but in Python over Chroma. This rung keeps the same retrieval idea and changes the data plane.

Fusion moves into the database. Dense and sparse both run in one SQL statement next to the data, so there is no per query index rebuild in Python. The naive baseline rebuilt a BM25 index over the whole corpus on every query; this removes that.

It also adds streaming answers over server sent events, conversational memory keyed by session id, asynchronous indexing through a Celery worker so uploads return immediately, and a labeled golden set that runs through the real retriever against the live database.

## Architecture

```mermaid
graph TD
    User[User in browser] --> UI[Streamlit chat UI]
    UI -->|X-API-Key| API[FastAPI, v1 router]
    API --> Chain[RAG chain, LCEL, streaming]
    API --> Memory[(application_logs, conversation memory)]
    Chain --> Retriever[HybridRetriever]
    Retriever --> PG[(Postgres and pgvector, langchain_pg_embedding)]
    Retriever --> Reranker[ReRankingRetriever, bge-reranker-v2-m3]
    Chain --> LLM[LLM: OpenAI, Anthropic or Ollama, chosen by model name]
    API --> Queue[Celery via Redis]
    Queue --> Index[index_document]
    Index --> PG
```

One database serves two roles: vectors, in `langchain_pg_collection` and `langchain_pg_embedding` written by PGVector, and memory plus catalog, in `application_logs` and `document_store` written by `src/api/db_utils.py`.

## Ingestion pipeline

How a document gets in.

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant R as Redis
    participant W as Celery worker
    participant PG as pgvector
    U->>A: POST /v1/upload-doc
    A->>A: sanitize filename, write temp file
    A->>PG: insert into document_store, returns file_id
    A->>R: enqueue process_document(path, filename)
    A-->>U: 202 and task_id, returns immediately
    W->>R: pick up task
    W->>W: load .pdf, .docx, .html or .txt
    W->>W: RecursiveCharacterTextSplitter(1000, overlap 200)
    W->>W: embed each chunk, task_type retrieval_document
    W->>PG: add_documents, file_id on every chunk
    U->>A: GET /v1/task/{id} for status
```

The detail that matters: every chunk carries its `file_id` in metadata, which is what lets `POST /v1/delete-doc` remove a document and all of its chunks. Indexing is idempotent per file_id.

## Retrieval pipeline

How a question finds context. This is the heart of this rung: one SQL query does dense, sparse and fusion together.

```mermaid
graph LR
    Q[Question] --> QE[Embed, task_type retrieval_query]
    QE --> SQL{{Single SQL statement}}
    Q --> SQL
    SQL --> D[dense CTE, cosine distance, row_number gives rank_dense]
    SQL --> S[sparse CTE, to_tsvector matches plainto_tsquery, ts_rank gives rank_sparse]
    D --> F[FULL OUTER JOIN on id]
    S --> F
    F --> RRF[score is 1/&#40;k+rank_dense&#41; plus 1/&#40;k+rank_sparse&#41;]
    RRF --> TOP[ORDER BY score DESC LIMIT k]
    TOP --> RR[bge-reranker-v2-m3 cross encoder]
    RR --> C[top_n chunks into the prompt]
```

1. Dense. `e.embedding <=> %(qvec)s::vector` is pgvector cosine distance, and `row_number()` gives `rank_dense`, capped at `pool`.
2. Sparse. Postgres full text, `to_tsvector('english', document) @@ plainto_tsquery(...)`, ranked by `ts_rank` into `rank_sparse`.
3. Fusion. A `FULL OUTER JOIN` keeps a chunk found by either arm, then RRF scores it as `1/(rrf_k + rank_dense) + 1/(rrf_k + rank_sparse)` with missing ranks coalesced to zero. Being rank based, it never has to normalize two incomparable score scales.
4. Rerank. `ReRankingRetriever` runs the cross encoder over the fused candidates and truncates to `reranker_top_n`. It is lazily loaded and injectable, so the test suite runs without downloading the model or importing torch.

Embeddings are asymmetric. Documents are embedded with `task_type=retrieval_document` and queries with `retrieval_query`, through `get_document_embeddings()` and `get_query_embeddings()`. Gemini embeddings are trained for that asymmetry; Ollama ignores the task type.

## Memory

How follow up questions work.

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant M as application_logs
    participant R as Retriever
    participant L as LLM
    U->>A: POST /v1/chat with session_id and question
    A->>M: get_chat_history(session_id)
    alt history is empty, first turn
        A->>R: retrieve with the raw question
    else has history
        A->>L: contextualize into a standalone question
        L-->>A: rewritten question
        A->>R: retrieve with the rewritten question
    end
    R-->>A: top_n chunks
    A->>L: QA prompt with context and history
    L-->>A: stream tokens as server sent events
    A->>M: insert_application_logs(session_id, question, answer, model)
```

The store is `application_logs(id, session_id, user_query, gpt_response, model, created_at)` with an index on `session_id`. `get_chat_history()` replays it as alternating human and AI messages.

Reformulation matters because "and what about the second one?" is unretrievable on its own. The contextualize step rewrites it into a standalone question before retrieval, and is skipped on the first turn, which saves a model call on the most common case. Memory is per session id, server side, in Postgres rather than the browser, so a refresh keeps the thread.

## LLM

| Role | Default | Alternatives | Notes |
|---|---|---|---|
| Generation | chosen by model name | OpenAI, Anthropic, Ollama | the provider is inferred from the model string |
| Embeddings, documents | models/gemini-embedding-001, 768d | Ollama nomic-embed-text | task_type retrieval_document |
| Embeddings, query | same model | same | task_type retrieval_query |
| Reranker | BAAI/bge-reranker-v2-m3 | disable with USE_RERANKER=false | local cross encoder, warmed at startup |

There are two prompts, both in `src/core/langchain_utils.py`: `CONTEXTUALIZE_SYSTEM` rewrites and never answers, and `QA_SYSTEM` answers only from context and says so rather than guessing. The chain is built on `langchain-core` LCEL rather than the legacy `langchain.chains` helpers, which is what keeps it stable across LangChain majors.

Running fully offline is a first class path: `EMBEDDING_PROVIDER=ollama` with a local `llama3.2:3b` and `USE_RERANKER=false` runs the whole system with no paid key.

## Features

| Area | Capability |
|---|---|
| Retrieval | Dense pgvector and sparse Postgres full text, fused with RRF in one SQL query |
| Reranking | Cross encoder bge-reranker-v2-m3, lazily loaded and injectable |
| Embeddings | Google gemini-embedding-001 with asymmetric task types, or local Ollama |
| Generation | OpenAI, Anthropic, or local Ollama, chosen by model name |
| Memory | Multi turn sessions in Postgres, with question reformulation |
| Streaming | Server sent events on the chat endpoint |
| Async indexing | Celery worker backed by Redis, with task status |
| Security | API key auth, rate limiting, input sanitization, CORS |
| Observability | Prometheus metrics at /metrics, structured logging |
| Evaluation | Retrieval metrics on a labeled golden set against the live database |
| Packaging | Docker Compose for the full stack, 42 tests, 86 percent coverage |

## Quick start

Docker Compose, full stack:

```bash
cp .env.example .env
# edit .env: set GOOGLE_API_KEY and one LLM key, or configure Ollama for a local run
make stack-up          # postgres, redis, api, worker, streamlit
open http://localhost:8501     # chat UI; API docs at :8000/docs
```

Fully offline, no paid keys:

```bash
make db-up                                   # postgres with pgvector, redis
ollama serve & ollama pull nomic-embed-text && ollama pull llama3.2:3b
make install
EMBEDDING_PROVIDER=ollama USE_RERANKER=false make dev   # API on :8000
make worker            # celery worker, second terminal
make frontend          # streamlit on :8501, third terminal
```

Upload a document in the sidebar, choose `llama3.2:3b`, and ask. The answer streams back grounded in your document.

## Sample data

Sample documents ship in [sample_data](sample_data): an HR handbook, a product FAQ, and a real SEC 10-K excerpt.

```bash
make load-samples
```

Then ask the questions in [sample_data/README.md](sample_data/README.md) and compare against the expected answers, including a memory follow up and an honesty check where it should decline rather than guess.

## Configuration

Environment variables, with optional profiles in `configs/dev.yml` and `configs/prod.yml`. Environment always wins.

| Setting | Default | Meaning |
|---|---|---|
| DATABASE_URL | postgresql://ragflow:ragflow@localhost:5432/ragflowpro | Postgres, vectors and memory |
| REDIS_URL | redis://localhost:6379/0 | Celery broker |
| EMBEDDING_PROVIDER | google | google or ollama |
| EMBEDDING_MODEL | models/gemini-embedding-001 | Google embedding model, 768d |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 | Cross encoder |
| USE_RERANKER | true | Turn reranking on or off |
| TOP_K | 5 | Candidates kept after fusion |
| RERANKER_TOP_N | 5 | Chunks kept after reranking |
| CHUNK_SIZE, CHUNK_OVERLAP | 1000, 200 | Splitter settings |
| API_KEY | change_me | Required in the X-API-Key header |

## API reference

| Method and path | Purpose |
|---|---|
| GET /health | Liveness, no auth |
| POST /v1/chat | Streaming RAG answer with conversation memory |
| POST /v1/upload-doc | Upload and asynchronously index a document |
| GET /v1/list-docs | List indexed documents |
| POST /v1/delete-doc | Delete a document and all of its chunks |
| GET /v1/task/{task_id} | Status of an async indexing task |
| GET /metrics | Prometheus metrics |

## Evaluation

Quality is measured, not assumed. `python -m eval.run`, or `make eval`, pushes a labeled golden set through the real hybrid retriever against the live database and reports retrieval metrics.

Latest run, 8 questions, local Ollama nomic-embed-text, k of 5:

| Metric | Value | Meaning |
|---|---|---|
| Top-1 accuracy | 1.000 | The top retrieved document is correct for every question |
| Hit@5 | 1.000 | The correct document is in the top 5 every time |
| MRR | 1.000 | The correct document is ranked first every time |
| Recall@5 | 1.000 | Every relevant document is retrieved |
| Precision@5 | 0.200 | A ceiling, not a defect: one relevant document over k of 5 |
| F1@5 | 0.333 | Harmonic mean of the above |

Read these honestly. They measure retrieval, not answer faithfulness. The metrics are computed in `eval/metrics.py` as hit@k, precision@k, recall@k, F1 and reciprocal rank, not by a model judge. It is a small hand labeled set on local embeddings, so treat it as a working baseline and rerun with production models for final figures. Answer level scoring such as faithfulness and answer relevancy is not implemented here; `ragas` is present in `requirements.txt` but unused, so wire it in the container and pin it, since it pulls packages with an open advisory.

## Testing

```bash
make test        # unit tests
make test-int    # integration tests, requires make db-up
```

42 tests, 86 percent coverage. Integration tests run against live Postgres, pgvector and Ollama, so hybrid retrieval, the pgvector round trip and the full chat pipeline are verified end to end rather than mocked.

## Project structure

```
src/api/          FastAPI app, endpoints, security, Postgres memory
src/core/         config, RAG chain (LCEL), logging
src/embeddings/   pgvector store, asymmetric embedding providers
src/retrieval/    hybrid retriever (RRF in SQL) and cross encoder reranker
src/worker/       Celery app and indexing task
frontend/         Streamlit chat UI
eval/             golden set and retrieval metrics harness
tests/            unit and integration tests
docker/           Dockerfile and Compose stack
configs/          dev and prod profiles
```

## The RAG line

This repo is the Modular (2023) rung. Each rung adds one idea and keeps the ones below it.

| Year | Repository | Strategy |
|---|---|---|
| 2022 | [rag-naive-2022](https://github.com/mlvpatel/rag-naive-2022) | Naive: one dense search over Chroma |
| 2023 | [rag-advanced-2023](https://github.com/mlvpatel/rag-advanced-2023) | Advanced: hybrid, RRF and cross encoder, in Python |
| 2023 | rag-modular-2023, this repo | Modular: pgvector, RRF in SQL, streaming, memory, evaluation |
| 2024 | [rag-graph-2024](https://github.com/mlvpatel/rag-graph-2024) | Graph: entity and triple knowledge graph linked into answers |
| 2024 | [rag-cache-2024](https://github.com/mlvpatel/rag-cache-2024) | Cache: no retrieval, corpus in context with a semantic cache |
| 2025 | [rag-agentic-2025](https://github.com/mlvpatel/rag-agentic-2025) | Agentic: bounded self correcting loop, confidence gated |
| 2026 | [rag-multiagent-2026](https://github.com/mlvpatel/rag-multiagent-2026) | Multi agent: supervisor, specialists, verifier |
| 2026 | [rag-multimodal-2026](https://github.com/mlvpatel/rag-multimodal-2026) | Multimodal: text and images in one vector space |

Every implementation is measured on the same golden questions, keyless, in the [rag-catalog](https://github.com/mlvpatel/rag-catalog) hub. To pick the right rung for a real problem, see [rag-ladder](https://github.com/mlvpatel/rag-ladder).

## Author

Malav Patel. GitHub @mlvpatel.

## License

Released under the MIT License. See [LICENSE](LICENSE).
