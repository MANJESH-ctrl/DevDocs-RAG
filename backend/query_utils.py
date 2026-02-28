from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
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

_embeddings = None
_llm        = None
_pc_index   = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)
    return _llm


def get_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = pc.Index(INDEX_NAME)
    return _pc_index


def hybrid_retriever(query: str, session_id: str, top_k: int = TOP_K, final_k: int = FINAL_K) -> List[Document]:
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
        if match["score"] < 0.3:
            continue
        meta = match["metadata"]
        text = meta.pop("text", "")
        docs.append(Document(page_content=text, metadata=meta))

    return docs[:final_k]


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        headers = " > ".join(v for k, v in doc.metadata.items() if k.startswith("Header"))
        parts.append(f"Source: {source} | {headers}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


TEMPLATE = """You are an expert on developer documentation.
Answer the question using only the provided context.
Always cite the source file and section headers in your answer.

Context:
{context}

Question: {question}

Answer:"""


def get_rag_chain(session_id: str):
    prompt = PromptTemplate.from_template(TEMPLATE)

    def _retrieve(q: str) -> str:
        docs = hybrid_retriever(q, session_id)
        return format_docs(docs)

    chain = (
        {
            "context": _retrieve,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain
