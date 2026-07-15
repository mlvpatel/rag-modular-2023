"""Run the rag-modular-2023 retrieval evaluation against the live database.

Indexes the golden documents into a dedicated collection using the configured
embeddings (Ollama by default here, for a keyless run), retrieves for every
golden question with the real hybrid retriever, and reports precision, recall,
F1, Hit@k, and MRR. Run with: python -m eval.run
"""

import psycopg
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from eval.golden import DOCUMENTS, QUESTIONS
from eval.metrics import (
    f1_score,
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.core.config import settings
from src.embeddings.vectorstore_utils import _sqlalchemy_url
from src.retrieval.retrievers import HybridRetriever

COLLECTION = "rag_modular_eval"


def _cleanup():
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM langchain_pg_embedding e "
                "USING langchain_pg_collection c "
                "WHERE e.collection_id = c.uuid AND c.name = %s",
                (COLLECTION,),
            )


def main(k: int = None) -> dict:
    k = k or settings.top_k
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model, base_url=settings.ollama_base_url
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=_sqlalchemy_url(),
        use_jsonb=True,
    )
    _cleanup()
    store.add_documents(
        [
            Document(page_content=text, metadata={"file_id": fid})
            for fid, text in DOCUMENTS.items()
        ]
    )
    retriever = HybridRetriever(embeddings=embeddings, collection=COLLECTION, k=k)

    precisions, recalls, f1s, hits, rrs, top1 = [], [], [], [], [], []
    for question, relevant in QUESTIONS:
        ranked = [doc.metadata.get("file_id") for doc in retriever.invoke(question)]
        p = precision_at_k(ranked, relevant, k)
        r = recall_at_k(ranked, relevant, k)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1_score(p, r))
        hits.append(hit_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
        top1.append(precision_at_k(ranked, relevant, 1))

    _cleanup()

    results = {
        "k": k,
        "questions": len(QUESTIONS),
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1s),
        "hit_rate": mean(hits),
        "mrr": mean(rrs),
        "top1_accuracy": mean(top1),
    }

    print(
        f"rag-modular-2023 retrieval evaluation (k={k}, {results['questions']} questions)"
    )
    print(
        f"  Top-1 accuracy: {results['top1_accuracy']:.3f}   (top result is the right document)"
    )
    print(
        f"  Hit@{k}:        {results['hit_rate']:.3f}   (right document in the top {k})"
    )
    print(f"  MRR:            {results['mrr']:.3f}")
    print(f"  Recall@{k}:     {results['recall']:.3f}")
    print(
        f"  Precision@{k}:  {results['precision']:.3f}   (bounded by one relevant doc over {k})"
    )
    print(f"  F1@{k}:         {results['f1']:.3f}")
    return results


if __name__ == "__main__":
    main()
