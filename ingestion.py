import os
import pickle
from uuid import uuid4
from typing import List, Callable, Optional

import torch
torch.set_num_threads(4)

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from config import (
    PINECONE_API_KEY, INDEX_NAME, CLOUD_REGION, EMBEDDING_MODEL,
    CHUNK_SIZE, BATCH_SIZE,
    MD_SPLITTER, ENCODING, TOKEN_SPLITTER,
)

# ── Lazy singletons ───────────────────────────────────────────
_embeddings = None
_pc_index   = None
_bm25_store = {}   # session_id -> fitted BM25Encoder in RAM

# ── BM25 disk cache dir ───────────────────────────────────────
BM25_CACHE_DIR = "/tmp/bm25_cache"
os.makedirs(BM25_CACHE_DIR, exist_ok=True)

def _bm25_path(session_id: str) -> str:
    return os.path.join(BM25_CACHE_DIR, f"{session_id}.pkl")


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings


def get_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=CLOUD_REGION),
            )
        _pc_index = pc.Index(INDEX_NAME)
    return _pc_index


def get_bm25_for_session(session_id: str) -> BM25Encoder:
    if session_id in _bm25_store:
        return _bm25_store[session_id]
    # try disk — survives server restarts
    path = _bm25_path(session_id)
    if os.path.exists(path):
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        _bm25_store[session_id] = bm25
        return bm25
    raise ValueError(
        f"BM25 not found for session {session_id}. "
        "Server may have restarted — please re-upload."
    )


# ── PDF → markdown (pymupdf4llm — fast, no system deps) ───────
def pdf_to_markdown(filepath: str) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(filepath).strip()


# ── Chunking ──────────────────────────────────────────────────
def hierarchical_split(documents: List[Document]) -> List[Document]:
    final_chunks: List[Document] = []
    for doc in documents:
        for md_chunk in MD_SPLITTER.split_text(doc.page_content):
            content = (md_chunk.page_content or "").strip()
            if not content:
                continue
            metadata = {**doc.metadata, **md_chunk.metadata}
            if len(content) < CHUNK_SIZE * 3:
                final_chunks.append(Document(page_content=content, metadata=metadata))
                continue
            tokens = ENCODING.encode(content)
            if len(tokens) > CHUNK_SIZE:
                sub_docs = TOKEN_SPLITTER.split_documents(
                    [Document(page_content=content, metadata=metadata)]
                )
                final_chunks.extend(sub_docs)
            else:
                final_chunks.append(Document(page_content=content, metadata=metadata))
    return [
        c for c in final_chunks
        if c.page_content and isinstance(c.page_content, str) and len(c.page_content.strip()) >= 50
    ]


# ── Main ingest pipeline ──────────────────────────────────────
def ingest_document(
    file_path: str,
    session_id: str,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    def step(msg: str):
        if progress_cb:
            progress_cb(msg)

    step("Parsing PDF...")
    md_text = pdf_to_markdown(file_path)
    docs = [Document(
        page_content=md_text,
        metadata={"source_file": os.path.basename(file_path)},
    )]

    step("Chunking document...")
    chunks = hierarchical_split(docs)
    if not chunks:
        raise ValueError("No valid chunks extracted from PDF.")
    texts = [c.page_content for c in chunks]

    step(f"Embedding {len(chunks)} chunks...")
    embeddings = get_embeddings()
    dense_vecs = embeddings.embed_documents(texts)

    step("Building keyword index...")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    _bm25_store[session_id] = bm25
    with open(_bm25_path(session_id), "wb") as f:
        pickle.dump(bm25, f)

    step("Preparing vectors...")
    vectors = []
    for i, chunk in enumerate(chunks):
        sparse_vec = bm25.encode_documents(chunk.page_content)
        meta = chunk.metadata.copy()
        meta["text"] = chunk.page_content
        vectors.append({
            "id": str(uuid4()),
            "values": dense_vecs[i],
            "sparse_values": sparse_vec,
            "metadata": meta,
        })

    step("Uploading to Pinecone...")
    index = get_index()
    if len(vectors) <= BATCH_SIZE:
        index.upsert(vectors=vectors, namespace=session_id)
    else:
        for i in range(0, len(vectors), BATCH_SIZE):
            index.upsert(vectors=vectors[i: i + BATCH_SIZE], namespace=session_id)
