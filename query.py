from typing import List, Optional, AsyncGenerator

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder

from config import GROQ_API_KEY, LLM_MODEL, TOP_K, FINAL_K
from ingestion import get_embeddings, get_index, get_bm25_for_session

_llm      = None
_reranker = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)
    return _llm


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ── Hybrid retriever + CrossEncoder reranker ─────────────────
def hybrid_retriever(
    query: str,
    session_id: str,
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
) -> List[Document]:
    embeddings = get_embeddings()
    bm25       = get_bm25_for_session(session_id)
    index      = get_index()

    dense_vec  = embeddings.embed_query(query)
    sparse_vec = bm25.encode_queries(query)

    result = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        namespace=session_id,
    )

    docs: List[Document] = []
    for match in result["matches"]:
        if match["score"] < 0.2:
            continue
        meta = dict(match["metadata"])
        text = meta.pop("text", "")
        docs.append(Document(page_content=text, metadata=meta))

    if not docs:
        return docs

    reranker = get_reranker()
    pairs    = [[query, doc.page_content] for doc in docs]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:final_k]]


# ── Helpers ───────────────────────────────────────────────────
def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source  = doc.metadata.get("source_file", "unknown")
        headers = " > ".join(v for k, v in doc.metadata.items() if k.startswith("Header"))
        parts.append(f"[Source: {source}{' | ' + headers if headers else ''}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


TEMPLATE = """You are an expert assistant for developer documentation.
Use the conversation history to understand follow-up questions and references like "my first query", "that", "it", "explain again", etc.
Answer using ONLY the provided document context. Always cite the source file and section headers. Give as much information as possible regarding the query which satisfies query as answer.
the answers should be very helpful and detailed, and should be based on the provided document context. If the question is not answerable using the provided document context, say "I don't know" and do not attempt to fabricate an answer.
focus on code more if present in the document context, and explain it in detail. If the question is about a specific section of the document, focus on that section and its related sections. If the question is about a specific term or concept, look for it in the document context and explain it in detail.


=== CONVERSATION HISTORY ===
{history}

=== DOCUMENT CONTEXT ===
{context}

=== CURRENT QUESTION ===
{question}

Answer:"""


# ── Streaming RAG (main path) ─────────────────────────────────
async def stream_rag_response(question: str, session_id: str, history: list = []):
    """Async generator — yields SSE-ready dicts: sources → tokens → done."""
    docs = hybrid_retriever(question, session_id)

    sources = []
    for doc in docs:
        source  = doc.metadata.get("source_file", "unknown")
        headers = " > ".join(v for k, v in doc.metadata.items() if k.startswith("Header"))
        sources.append({"source": source, "section": headers, "preview": doc.page_content[:150]})

    yield {"type": "sources", "sources": sources}

    prompt      = PromptTemplate.from_template(TEMPLATE)
    history_str = format_history(history)
    context_str = format_docs(docs)

    chain = (
        {
            "context":  lambda _: context_str,
            "history":  lambda _: history_str,
            "question": RunnablePassthrough(),
        }
        | prompt | get_llm() | StrOutputParser()
    )

    async for chunk in chain.astream(question):
        yield {"type": "token", "content": chunk}

    yield {"type": "done"}
