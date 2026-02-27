# backend/query_utils.py
from typing import List

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K,
    FINAL_K,
)
from ingestion_utils import get_bm25_for_session

# ---------- singletons ----------
_embeddings = None
_llm        = None
_pc_index   = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0,
            groq_api_key=GROQ_API_KEY,
        )
    return _llm


def get_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = pc.Index(INDEX_NAME)
    return _pc_index


# ---------- hybrid retriever ----------
def hybrid_retriever(
    query: str, session_id: str, top_k: int = TOP_K, final_k: int = FINAL_K
) -> List[Document]:
    embeddings = get_embeddings()
    bm25: BM25Encoder = get_bm25_for_session(session_id)
    index = get_index()

    dense_vec = embeddings.embed_query(query)
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
        if match["score"] < 0.3:   # simple quality filter
            continue
        metadata = match["metadata"]
        page_content = metadata.pop("text", "")
        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs[:final_k]


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        headers = " > ".join(
            v for k, v in doc.metadata.items() if k.startswith("Header")
        )
        formatted.append(
            f"Source: {source} | {headers}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


TEMPLATE = """You are an expert on developer documentation.
Answer the question using only the provided context.
Always cite the source file and section headers in your answer.

Context:
{context}

Question: {question}

Answer:"""


def get_rag_chain(session_id: str, history_text: str = "None"):
    prompt = PromptTemplate.from_template(TEMPLATE)   # TEMPLATE now has {history}

    def _retrieve(q: str) -> str:
        docs = hybrid_retriever(q, session_id)
        return format_docs(docs)

    chain = (
        {
            "context":  _retrieve,
            "question": RunnablePassthrough(),
            "history":  lambda _: history_text
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain