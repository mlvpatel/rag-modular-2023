"""Background tasks: asynchronous document indexing into pgvector."""
import logging

from src.api.db_utils import delete_document_record, insert_document_record
from src.embeddings.vectorstore_utils import index_document
from src.worker.celery_app import celery_app

logger = logging.getLogger("ragflowpro")


@celery_app.task(name="process_document")
def process_document(file_path: str, filename: str) -> dict:
    """Index a document.

    Ordering matters: the database record is inserted first so we get a real
    integer file_id, which is then attached to every chunk stored in pgvector.
    If indexing fails, the record is rolled back so we never leave a document
    listed as present with nothing indexed behind it.
    """
    file_id = insert_document_record(filename)
    indexed = index_document(file_path, file_id)
    if not indexed:
        delete_document_record(file_id)
        logger.error("Indexing failed for %s, rolled back record %s", filename, file_id)
        return {"status": "failed", "file_id": file_id}
    return {"status": "completed", "file_id": file_id}
