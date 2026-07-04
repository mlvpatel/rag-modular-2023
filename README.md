# RagFlowPro

**Retrieval augmented generation chatbot with hybrid retrieval, streaming answers, and a measurable quality gate.**

RagFlowPro answers questions about your own documents. It combines dense vector search and sparse keyword search inside the database, streams grounded answers token by token, remembers the conversation, and can run fully locally at no cost or against cloud models in production.

![RagFlowPro answering a document question live on a local model](assets/videos/ragflowpro-demo.gif)

The animation above is a live, unedited run. The model is a local llama3.2, the documents (including a real SEC 10-K) are indexed in pgvector, and the answer streams in grounded in them. No paid keys were used.

**Full recording and stills:** the complete screen recording of all four questions is at [assets/videos/ragflowpro-demo.webm](assets/videos/ragflowpro-demo.webm), and a full resolution screenshot is at [assets/screenshots/ragflowpro-sample-data-demo.png](assets/screenshots/ragflowpro-sample-data-demo.png).

[![CI](https://github.com/mlvpatel/RagFlowPro/actions/workflows/ci.yml/badge.svg)](https://github.com/mlvpatel/RagFlowPro/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) ![Postgres](https://img.shields.io/badge/Postgres-pgvector-blue) ![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

## Why RagFlowPro

- **Hybrid retrieval that actually scales.** Dense pgvector search and sparse Postgres full text search are fused with Reciprocal Rank Fusion inside a single SQL query. There is no per query index rebuild, so retrieval stays fast as the corpus grows.
- **Streaming answers.** The chat endpoint streams tokens as server sent events, so the user sees the answer form in real time instead of waiting for a full response.
- **Conversational memory.** Every turn is stored in Postgres under a session id, so follow up questions resolve against the earlier conversation.
- **Runs local or cloud.** A provider switch lets the same code run on Google gemini-embedding-001 with a cloud LLM, or fully offline on Ollama models, at no cost.
- **Measurable quality.** A RAGAS evaluation harness measures precision, recall, and faithfulness, so answer quality is a number you can track, not a claim.

## Features

| Area | Capability |
|---|---|
| Retrieval | Dense pgvector plus sparse Postgres full text, fused with RRF in one SQL query |
| Reranking | Cross encoder bge-reranker-v2-m3, warmed at startup |
| Embeddings | Google gemini-embedding-001, or local Ollama nomic-embed-text |
| Generation | OpenAI, Anthropic, or local Ollama, chosen by model name |
| Memory | Multi turn sessions stored in Postgres |
| Streaming | Server sent events on the chat endpoint |
| Async indexing | Celery worker backed by Redis |
| Security | API key auth, rate limiting, input sanitization, CORS |
| Observability | Prometheus metrics at /metrics, structured logging |
| Evaluation | RAGAS harness for precision, recall, faithfulness |
| Packaging | Docker Compose for the full stack, 42 tests, 86 percent coverage |

## Architecture

```mermaid
graph TD
    User[User in browser] --> UI[Streamlit chat UI]
    UI --> API[FastAPI backend, v1 router]
    API --> Chain[RAG chain, LCEL, streaming]
    API --> Memory[(Postgres conversation memory)]
    Chain --> Retriever[Hybrid retriever]
    Retriever --> PG[(Postgres with pgvector)]
    Retriever --> Reranker[bge-reranker-v2-m3]
    Chain --> LLM[LLM provider OpenAI, Anthropic, or Ollama]
    API --> Queue[Celery worker via Redis]
    Queue --> Index[Index documents into pgvector]
    Index --> PG
    Embed[Embeddings gemini-embedding-001 or Ollama] --> PG
```

## How a question is answered

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit UI
    participant A as FastAPI
    participant M as Postgres memory
    participant R as Hybrid retriever
    participant L as LLM
    U->>S: Types a question
    S->>A: POST /v1/chat (streaming)
    A->>M: Read conversation history
    A->>R: Reformulate if history, then retrieve
    R->>R: Dense pgvector plus sparse FTS, fused with RRF in SQL
    R->>A: Top ranked chunks
    A->>L: Prompt with the retrieved context
    L-->>A: Answer tokens
    A-->>S: Stream tokens as server sent events
    A->>M: Persist the turn for memory
```

## Tech stack

Python 3.11, FastAPI, Streamlit, LangChain core (LCEL), Postgres with the pgvector extension, psycopg 3, sentence-transformers, Celery, Redis, Prometheus, RAGAS. Embeddings default to Google gemini-embedding-001, with Ollama as a local option.

## Quick start

### Option A, Docker Compose (full stack)

```bash
cp .env.example .env
# edit .env: set GOOGLE_API_KEY and one LLM key, or configure Ollama for a local run
make stack-up          # builds and starts postgres, redis, api, worker, streamlit
open http://localhost:8501     # the chat UI
# API docs at http://localhost:8000/docs
```

### Option B, local development with a fully offline model (no paid keys)

```bash
# 1. Start the data services
make db-up             # postgres with pgvector, plus redis

# 2. Start Ollama and pull the local models
ollama serve &
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# 3. Install and run
make install
EMBEDDING_PROVIDER=ollama USE_RERANKER=false make dev        # API on :8000
make worker            # celery worker, in a second terminal
make frontend          # streamlit on :8501, in a third terminal
```

Upload a document in the sidebar, choose the llama3.2:3b model, and ask a question. The answer streams back grounded in your document.

## Try it with the bundled sample data

The repo ships with sample documents in [sample_data](sample_data), an HR handbook, a product FAQ, and a real SEC 10-K excerpt, so you can run and evaluate the system without supplying your own files. With the stack up, load them in one command:

```bash
make load-samples
```

Then ask the questions listed in [sample_data/README.md](sample_data/README.md) and compare the answers to the expected ones, including a memory follow up and an honesty check where it should decline to answer rather than guess.

## Configuration

Configuration comes from environment variables, with optional profiles in `configs/dev.yml` and `configs/prod.yml`. Environment variables always take precedence.

| Setting | Default | Meaning |
|---|---|---|
| DATABASE_URL | postgresql://ragflow:ragflow@localhost:5432/ragflowpro | Postgres, memory and vectors |
| REDIS_URL | redis://localhost:6379/0 | Celery broker |
| EMBEDDING_PROVIDER | google | google or ollama |
| EMBEDDING_MODEL | models/gemini-embedding-001 | Google embedding model |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 | Cross encoder reranker |
| USE_RERANKER | true | Turn reranking on or off |
| TOP_K | 5 | Candidates retrieved before reranking |
| API_KEY | change_me | Required in the X-API-Key header |

## API reference

| Method and path | Purpose |
|---|---|
| GET /health | Liveness, no auth |
| POST /v1/chat | Streaming RAG answer with conversation memory |
| POST /v1/upload-doc | Upload and asynchronously index a document |
| GET /v1/list-docs | List indexed documents |
| POST /v1/delete-doc | Delete a document and its chunks |
| GET /v1/task/{task_id} | Status of an async indexing task |
| GET /metrics | Prometheus metrics |

## Evaluation

Quality is measured, not assumed. The harness (`python -m eval.run`, or `make eval`) runs a labeled golden question set through the real hybrid retriever against the live database and reports retrieval metrics.

Latest run, 8 questions, local Ollama nomic-embed-text embeddings, k of 5:

| Metric | Value | Meaning |
|---|---|---|
| Top-1 accuracy | 1.000 | The top retrieved document is the correct one for every question |
| Hit@5 | 1.000 | The correct document is in the top 5 for every question |
| MRR | 1.000 | The correct document is ranked first every time |
| Recall@5 | 1.000 | Every relevant document is retrieved |
| Precision@5 | 0.200 | Bounded above by one relevant document over k of 5, so this is the ceiling here, not a defect |
| F1@5 | 0.333 | Harmonic mean of the precision and recall above |

Retrieval is effectively perfect on this set: the right document is ranked first every time. These numbers use local embeddings on a small hand labeled set, so treat them as a working baseline and rerun with the production models for the final figures. A full answer quality evaluation (faithfulness and answer relevancy judged by an LLM) can be added with RAGAS; ragas pulls packages with an open advisory, so pin it and run it in the container.

## Testing

```bash
make test        # unit tests
make test-int    # integration tests (requires make db-up)
```

The suite has 42 tests. Coverage is 86 percent. Integration tests run against live Postgres, pgvector, and Ollama, so the hybrid retrieval, the pgvector round trip, and the full chat pipeline are verified end to end, not mocked.

## Project structure

```
src/api/          FastAPI app, endpoints, security, Postgres memory
src/core/         config, RAG chain, logging
src/embeddings/   pgvector store and embedding providers
src/retrieval/    hybrid retriever and reranker
src/worker/       Celery app and indexing task
frontend/         Streamlit chat UI
eval/             RAGAS evaluation harness
tests/            unit and integration tests
docker/           Dockerfile and Compose stack
configs/          dev and prod profiles
```

## Author

Malav Patel. GitHub @mlvpatel.

## License

Released under the MIT License. See [LICENSE](LICENSE). MIT is a deliberate choice for this project: it is the simplest and most permissive of the common open source licenses, so anyone, including a client evaluating the work, can read, run, modify, and reuse the code with no friction and no legal overhead.
