"""
RAG chain for RagFlowPro, built directly on langchain-core (LCEL).

Building on langchain-core rather than the legacy langchain.chains helpers
keeps this stable across langchain major versions. The flow is:
reformulate the question using chat history (skipped on the first turn),
retrieve with the hybrid retriever, then generate an answer grounded in the
retrieved context. Both a single-shot and a streaming entry point are
provided. The LLM provider is chosen from the model name, so the same code
serves OpenAI, Anthropic, and local Ollama models.
"""
import logging
from typing import Any, Iterator, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.config import settings
from src.embeddings.vectorstore_utils import get_query_embeddings
from src.retrieval.retrievers import HybridRetriever, ReRankingRetriever

logger = logging.getLogger(__name__)

CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

QA_SYSTEM = (
    "You are a helpful assistant for RagFlowPro. Use the following retrieved "
    "context to answer the user question accurately and concisely. If the "
    "context does not contain enough information, say so rather than guessing.\n\n"
    "Context:\n{context}"
)

_contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _make_llm(model: str):
    """Return a chat model for the given model name, provider chosen by name."""
    name = model.lower()
    if any(tag in name for tag in ("llama", "qwen", "deepseek", "mistral", "gemma", "phi")):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url)
    if "claude" in name:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=settings.openai_api_key)


def get_final_retriever():
    """The hybrid retriever, wrapped in the cross encoder reranker when enabled."""
    base = HybridRetriever(embeddings=get_query_embeddings(), k=settings.top_k)
    if settings.use_reranker:
        return ReRankingRetriever(base_retriever=base, top_n=settings.reranker_top_n)
    return base


def warm_reranker() -> None:
    """Load the reranker at startup so the first chat request is not slow.

    Never fatal: if the reranker cannot load (for example torch is not
    installed in this environment), log a warning and continue.
    """
    if not settings.use_reranker:
        return
    try:
        from src.retrieval.retrievers import get_reranker

        get_reranker()
    except Exception as exc:
        logger.warning("Reranker warm-up skipped: %s", exc)


def _to_lc_messages(chat_history) -> List[Any]:
    """Convert stored {role, content} dicts into langchain message objects."""
    messages: List[Any] = []
    for turn in chat_history or []:
        if turn.get("role") in ("ai", "assistant"):
            messages.append(AIMessage(content=turn["content"]))
        else:
            messages.append(HumanMessage(content=turn["content"]))
    return messages


def _reformulate_query(llm, user_input: str, history: List[Any]) -> str:
    """Rewrite the question to be standalone, skipped when there is no history."""
    if not history:
        return user_input
    chain = _contextualize_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input, "chat_history": history})


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(model: str, user_input: str, chat_history=None) -> dict:
    """Run the full RAG flow once and return {answer, context}."""
    llm = _make_llm(model)
    retriever = get_final_retriever()
    history = _to_lc_messages(chat_history)
    query = _reformulate_query(llm, user_input, history)
    docs = retriever.invoke(query)
    chain = _qa_prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"input": user_input, "chat_history": history, "context": _format_context(docs)}
    )
    return {"answer": answer, "context": docs}


def stream_answer(model: str, user_input: str, chat_history=None) -> Iterator[str]:
    """Stream the answer tokens for the RAG flow (retrieval runs first)."""
    llm = _make_llm(model)
    retriever = get_final_retriever()
    history = _to_lc_messages(chat_history)
    query = _reformulate_query(llm, user_input, history)
    docs = retriever.invoke(query)
    chain = _qa_prompt | llm
    for chunk in chain.stream(
        {"input": user_input, "chat_history": history, "context": _format_context(docs)}
    ):
        text = getattr(chunk, "content", "")
        if text:
            yield text
