"""RagFlowPro API: streaming RAG chat plus document management."""
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.db_utils import (
    delete_document_record,
    get_all_documents,
    get_chat_history,
    init_db,
    insert_application_logs,
)
from src.api.pydantic_models import DeleteFileRequest, DocumentInfo, QueryInput
from src.api.security import limiter, sanitize_question, verify_api_key
from src.core.config import settings
from src.core.langchain_utils import stream_answer, warm_reranker
from src.core.logging_config import configure_logging, logger
from src.embeddings.vectorstore_utils import delete_doc
from src.worker.celery_app import celery_app
from src.worker.tasks import process_document

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".html", ".txt", ".md"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        init_db()
    except Exception as exc:
        logger.warning("init_db skipped at startup: %s", exc)
    warm_reranker()
    yield


app = FastAPI(title="RagFlowPro API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

v1 = APIRouter(prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}


@v1.post("/chat")
@limiter.limit("60/minute")
def chat(request: Request, query: QueryInput, _: str = Depends(verify_api_key)):
    session_id = query.session_id or str(uuid.uuid4())
    question = sanitize_question(query.question)
    history = get_chat_history(session_id)

    def event_stream():
        parts = []
        for token in stream_answer(query.model, question, history):
            parts.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        answer = "".join(parts)
        try:
            insert_application_logs(session_id, question, answer, query.model)
        except Exception as exc:
            logger.error("Failed to persist chat turn: %s", exc)
        yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'model': query.model})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@v1.post("/upload-doc")
@limiter.limit("10/minute")
async def upload_doc(
    request: Request, file: UploadFile = File(...), _: str = Depends(verify_api_key)
):
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    upload_dir = os.getenv("UPLOAD_DIR", "data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, os.path.basename(file.filename))
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task = process_document.delay(path, file.filename)
    return {"task_id": task.id, "status": "processing", "filename": file.filename}


@v1.get("/list-docs", response_model=list[DocumentInfo])
def list_docs(_: str = Depends(verify_api_key)):
    return get_all_documents()


@v1.post("/delete-doc")
def delete_document(req: DeleteFileRequest, _: str = Depends(verify_api_key)):
    vector_ok = delete_doc(req.file_id)
    record_ok = delete_document_record(req.file_id)
    if not (vector_ok and record_ok):
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"status": "deleted", "file_id": req.file_id}


@v1.get("/task/{task_id}")
def task_status(task_id: str, _: str = Depends(verify_api_key)):
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.status}


app.include_router(v1)
