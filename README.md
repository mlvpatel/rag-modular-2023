# rag-modular-2023

Modular production RAG: hybrid retrieval fused in one SQL query, streaming answers, conversational memory, and a measured retrieval gate.

Part of the RAG line, a series of reference enterprise RAG implementations, one per retrieval strategy. This repository is the Modular (2023) rung. See [the line](#the-rag-line) below.

This service answers questions about your own documents. It runs dense vector search and sparse keyword search inside Postgres, fuses them with Reciprocal Rank Fusion in a single query, reranks with a cross encoder, streams grounded answers token by token, remembers the conversation, and runs fully locally at no cost or against cloud models in production.

![RagChatBot banner](assets/banner.svg)

The animation above is a live, unedited run. The model is a local llama3.2, the documents (including a real SEC 10-K) are indexed in pgvector, and the answer streams in grounded in them. No paid keys were used.

[![CI](https://github.com/mlvpatel/rag-modular-2023/actions/workflows/ci.yml/badge.svg)](https://github.com/mlvpatel/rag-modular-2023/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) ![Postgres](https://img.shields.io/badge/Postgres-pgvector-blue) ![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

## Contents

[What this rung adds](#what-this-rung-adds) · [Tech stack](#tech-stack) · [System context](#system-context-c4-level-1) · [Containers](#containers-c4-level-2) · [Data model](#data-model) · [Ingestion](#ingestion-pipeline) · [Retrieval](#retrieval-pipeline) · [The mathematics](#the-mathematics) · [Memory](#memory) · [LLM](#llm) · [Quick start](#quick-start) · [Configuration](#configuration) · [API](#api-reference) · [Evaluation](#evaluation) · [Testing](#testing) · [Structure](#project-structure)

## What this rung adds

The rung below, [rag-advanced-2023](https://github.com/mlvpatel/rag-advanced-2023), already does hybrid retrieval and reranking, but in Python over Chroma. This rung keeps the same retrieval idea and changes the data plane.

Fusion moves into the database. Dense and sparse both run in one SQL statement next to the data, so there is no per query index rebuild in Python. The naive baseline rebuilt a BM25 index over the whole corpus on every query; this removes that.

It also adds streaming answers over server sent events, conversational memory keyed by session id, asynchronous indexing through a Celery worker so uploads return immediately, and a labeled golden set that runs through the real retriever against the live database.

## Tech stack

| Layer | Choice | Why this one |
|---|---|---|
| API | FastAPI, Python 3.11 | async, native SSE streaming, generated OpenAPI docs |
| UI | Streamlit | a chat surface in a few files, no frontend build |
| Orchestration | langchain-core (LCEL) | composable chains without the legacy `langchain.chains` helpers, stable across majors |
| Vector store | Postgres 16 with pgvector | one database for vectors and memory, and fusion can run in SQL |
| Driver | psycopg 3 | the hybrid query is raw SQL, run on a connection pool |
| Sparse search | Postgres full text (`tsvector`) | no second engine to run or keep in sync |
| Reranker | bge-reranker-v2-m3 (sentence-transformers) | multilingual cross encoder, Apache-2.0, runs locally |
| Embeddings | gemini-embedding-001, or Ollama nomic-embed-text | cloud quality by default, keyless local path for development |
| Generation | OpenAI, Anthropic, or Ollama | chosen by model name, so the same code serves all three |
| Async work | Celery with Redis | uploads return immediately, indexing happens off the request path |
| Observability | Prometheus, structured logging | `/metrics` is scrapeable, logs are parseable |
| Packaging | Docker Compose | one command brings up the whole stack |

## System context (C4 level 1)

```mermaid
graph TB
    user([Analyst<br/>asks questions about internal documents])
    sys[rag-modular-2023<br/>Retrieval augmented QA service]
    llm[LLM provider<br/>OpenAI, Anthropic or Ollama]
    emb[Embedding provider<br/>Google or Ollama]
    user -->|asks a question, uploads a document| sys
    sys -->|grounded answer with sources| user
    sys -->|prompt with retrieved context| llm
    sys -->|text to embed| emb
```

The service is the only system that touches the documents. Both providers are swappable, and both have a local option, so the whole thing can run with no third party at all.

## Containers (C4 level 2)

```mermaid
graph TB
    subgraph client[Client]
        ui[Streamlit UI<br/>chat, upload, model picker]
    end
    subgraph svc[Service]
        api[FastAPI<br/>v1 router, auth, SSE]
        worker[Celery worker<br/>indexing]
    end
    subgraph data[Data]
        pg[(Postgres + pgvector<br/>vectors, memory, catalog)]
        redis[(Redis<br/>broker)]
    end
    ui -->|HTTPS| api
    api -->|enqueue| redis
    redis --> worker
    api -->|hybrid SQL, memory| pg
    worker -->|embed and insert chunks| pg
```

## Data model

One database, two concerns. PGVector owns the first two tables; `src/api/db_utils.py` owns the other two.

```mermaid
erDiagram
    langchain_pg_collection ||--o{ langchain_pg_embedding : contains
    document_store ||--o{ langchain_pg_embedding : "file_id in cmetadata"
    langchain_pg_collection {
        uuid uuid PK
        string name
    }
    langchain_pg_embedding {
        string id PK
        uuid collection_id FK
        vector embedding "768 dims"
        text document "chunk text"
        jsonb cmetadata "carries file_id"
    }
    document_store {
        int id PK
        string filename
        timestamp upload_timestamp
    }
    application_logs {
        int id PK
        text session_id "indexed"
        text user_query
        text gpt_response
        text model
        timestamp created_at
    }
```

`file_id` living in `cmetadata` on every chunk is what makes deletion work: removing a document means deleting its `document_store` row and every embedding whose metadata carries that id.

## Ingestion pipeline

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
    W->>W: embed chunks, task_type retrieval_document
    W->>PG: add_documents, file_id on every chunk
    U->>A: GET /v1/task/{id} for status
```

An upload returns before the work is done, so the task has a lifecycle worth watching:

```mermaid
stateDiagram-v2
    [*] --> PENDING: enqueued on Redis
    PENDING --> STARTED: worker picks it up
    STARTED --> SUCCESS: chunks embedded and stored
    STARTED --> FAILURE: parse or embed error
    SUCCESS --> [*]
    FAILURE --> [*]
```

`GET /v1/task/{task_id}` reports that state, so the UI can poll instead of blocking the upload request.

## Retrieval pipeline

One SQL query does dense, sparse and fusion together. Only the rerank happens in Python.

```mermaid
flowchart LR
    Q[Question] --> QE[Embed<br/>task_type retrieval_query]
    QE --> SQL{{Single SQL statement}}
    Q --> SQL
    SQL --> D["dense CTE<br/>cosine distance<br/>row_number gives rank_dense"]
    SQL --> S["sparse CTE<br/>tsvector matches tsquery<br/>ts_rank gives rank_sparse"]
    D --> F[FULL OUTER JOIN on id]
    S --> F
    F --> RRF[RRF score]
    RRF --> TOP[ORDER BY score DESC LIMIT k]
    TOP --> RR[bge-reranker-v2-m3<br/>cross encoder]
    RR --> C[top_n chunks into the prompt]
```

1. Dense. `e.embedding <=> %(qvec)s::vector` is pgvector cosine distance, and `row_number()` turns it into `rank_dense`, capped at `pool`.
2. Sparse. Postgres full text, `to_tsvector('english', document) @@ plainto_tsquery(...)`, ranked by `ts_rank` into `rank_sparse`.
3. Fusion. A `FULL OUTER JOIN` keeps a chunk found by either arm, and RRF combines the two ranks.
4. Rerank. `ReRankingRetriever` scores the fused candidates jointly with the query and truncates to `reranker_top_n`. It is lazily loaded and injectable, so the test suite runs without downloading the model or importing torch.

## The mathematics

**Embedding and cosine distance.** A chunk becomes a vector $\mathbf{v} \in \mathbb{R}^{768}$. pgvector's `<=>` operator is cosine *distance*, the complement of cosine similarity:

$$d_{\cos}(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}$$

so $d = 0$ is identical and $d = 2$ is opposite. Ordering ascending by `<=>` ranks the nearest chunk first.

**Why the embeddings are asymmetric.** A question and the passage answering it are not paraphrases, so embedding both with one function is a mismatch. Gemini takes a task type, giving two different maps $f_D$ for documents and $f_Q$ for queries trained so that

$$\text{sim}\big(f_Q(q),\, f_D(d)\big) \text{ is high when } d \text{ answers } q$$

rather than when $q$ and $d$ merely look alike. Hence `get_document_embeddings()` and `get_query_embeddings()`.

**Sparse ranking.** Postgres scores a document against a query with `ts_rank`, which weights matched lexeme frequency and position. Only rank order is used downstream, not the raw value.

**Reciprocal Rank Fusion.** Dense distance and `ts_rank` are different scales that cannot be compared or averaged. RRF avoids the problem by discarding the scores and using only ranks. For a chunk $d$ and retriever set $R$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

With $R = \{\text{dense}, \text{sparse}\}$, the implemented form is

$$\text{score}(d) = \frac{1}{k + \text{rank}_{\text{dense}}(d)} + \frac{1}{k + \text{rank}_{\text{sparse}}(d)}$$

where a chunk missing from one arm contributes $0$ for that term, which is exactly what `coalesce(..., 0)` does after the `FULL OUTER JOIN`. The constant $k$ (`rrf_k`, default 60) damps the top ranks: without it a rank-1 hit would score $1$ and a rank-2 hit $0.5$, so a single arm could dominate. With $k = 60$ those become $\tfrac{1}{61} \approx 0.0164$ and $\tfrac{1}{62} \approx 0.0161$, so agreement across both arms outweighs a strong opinion from one. This is the property that makes RRF robust, and it is why the fusion is rank based rather than a weighted score sum.

**Reranking.** The bi-encoder scores $q$ and $d$ independently, which is what makes indexing possible but costs accuracy. The cross encoder reads the pair together, $s = g(q \oplus d)$, and attends across both. It is too slow for the whole corpus and ideal for the $k$ survivors, which is why it sits last.

**Chunking.** With chunk size $c = 1000$ and overlap $o = 200$, the stride is $c - o = 800$, so a document of length $L$ yields roughly

$$n \approx \left\lceil \frac{L - o}{c - o} \right\rceil$$

chunks. The overlap exists so a fact spanning a boundary survives in at least one chunk.

**Retrieval metrics.** With $\mathcal{R}$ the relevant set and $\mathcal{T}_k$ the top $k$ retrieved:

$$P@k = \frac{|\mathcal{R} \cap \mathcal{T}_k|}{k} \qquad R@k = \frac{|\mathcal{R} \cap \mathcal{T}_k|}{|\mathcal{R}|} \qquad F_1 = \frac{2PR}{P+R}$$

$$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of the first relevant document for question $i$. Note $P@k$ is bounded by $|\mathcal{R}|/k$: with one relevant document and $k=5$, the best achievable precision is $0.2$, which is why the number below is a ceiling and not a defect.

## Memory

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

Reformulation matters because "and what about the second one?" is unretrievable on its own: its embedding carries no topic. The contextualize step rewrites it into a standalone question before retrieval, and is skipped on the first turn, which saves a model call on the most common case. Memory is per session id, server side, in Postgres rather than the browser, so a refresh keeps the thread.

## LLM

| Role | Default | Alternatives | Notes |
|---|---|---|---|
| Generation | chosen by model name | OpenAI, Anthropic, Ollama | the provider is inferred from the model string |
| Embeddings, documents | models/gemini-embedding-001, 768d | Ollama nomic-embed-text | task_type retrieval_document |
| Embeddings, query | same model | same | task_type retrieval_query |
| Reranker | BAAI/bge-reranker-v2-m3 | disable with USE_RERANKER=false | local cross encoder, warmed at startup |

There are two prompts, both in `src/core/langchain_utils.py`: `CONTEXTUALIZE_SYSTEM` rewrites and never answers, and `QA_SYSTEM` answers only from context and says so rather than guessing.

Running fully offline is a first class path: `EMBEDDING_PROVIDER=ollama` with a local `llama3.2:3b` and `USE_RERANKER=false` runs the whole system with no paid key.

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
| DATABASE_URL | postgresql://rag:rag@localhost:5432/rag_modular | Postgres, vectors and memory |
| REDIS_URL | redis://localhost:6379/0 | Celery broker |
| EMBEDDING_PROVIDER | google | google or ollama |
| EMBEDDING_MODEL | models/gemini-embedding-001 | Google embedding model, 768d |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 | Cross encoder |
| USE_RERANKER | true | Turn reranking on or off |
| TOP_K | 5 | Candidates kept after fusion |
| RERANKER_TOP_N | 5 | Chunks kept after reranking |
| CHUNK_SIZE, CHUNK_OVERLAP | 1000, 200 | Splitter settings |
| MAX_UPLOAD_MB | 25 | Uploads rejected above this size |

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

## A note on access

The service has no authentication, and that is a decision rather than an omission. It is a reference implementation meant to run on one machine: docker compose binds every published port, Postgres and Redis included, to `127.0.0.1`, and the containers run as a non-root user. A shipped default credential would be the worse option, since it reads as protection while sitting in a public repository. What remains is real: per route rate limiting, a hard size cap on uploads, HTML stripping on every question, and a narrow CORS origin. Put an authenticating gateway in front before exposing any of it beyond loopback.

## Evaluation

Quality is measured, not assumed. `python -m eval.run`, or `make eval`, pushes a labeled golden set through the real hybrid retriever against the live database and reports retrieval metrics.

Latest run, 8 questions, local Ollama nomic-embed-text, k of 5:

| Metric | Value | Meaning |
|---|---|---|
| Top-1 accuracy | 1.000 | The top retrieved document is correct for every question |
| Hit@5 | 1.000 | The correct document is in the top 5 every time |
| MRR | 1.000 | The correct document is ranked first every time |
| Recall@5 | 1.000 | Every relevant document is retrieved |
| Precision@5 | 0.200 | The ceiling for one relevant document at k of 5, not a defect |
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
docker/           Dockerfile, Compose stack, pgvector init SQL
configs/          dev and prod profiles
scripts/          sample data loader
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

## Author

Malav Patel. GitHub @mlvpatel.

## License

Released under the MIT License. See [LICENSE](LICENSE).
