# backend/ingestion_utils.py
import os
from uuid import uuid4
from typing import List

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import tiktoken

from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    CLOUD_REGION,
    EMBEDDING_MODEL,
    UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)

# ---------- singletons ----------
_embeddings = None
_pc_index   = None

# keep BM25 per session in memory for this process
_bm25_store = {}   # session_id -> BM25Encoder


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL
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


# ---------- PDF → markdown ----------
def pdf_to_markdown(filepath: str) -> str:
    print(f"[INGEST] Converting: {os.path.basename(filepath)}")
    elements = partition_pdf(
        filename=filepath,
        strategy="fast",          # optimized for speed
        infer_table_structure=False,
        languages=["eng"],
    )
    md_text = ""
    for el in elements:
        # prefer text_as_html if available
        if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
            md_text += f"\n{el.metadata.text_as_html}"
        elif el.text:
            md_text += f"\n{el.text}"
    return md_text.strip()


# ---------- chunking ----------
def hierarchical_split(documents: List[Document]) -> List[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ".", ""],
    )
    encoding = tiktoken.get_encoding("cl100k_base")

    final_chunks: List[Document] = []

    for doc in documents:
        md_chunks = md_splitter.split_text(doc.page_content)
        for md_chunk in md_chunks:
            content = (md_chunk.page_content or "").strip()
            if not content:
                continue

            metadata = {**doc.metadata, **md_chunk.metadata}
            tokens = encoding.encode(content)
            if len(tokens) > CHUNK_SIZE:
                temp_doc = Document(page_content=content, metadata=metadata)
                sub_chunks = token_splitter.split_documents([temp_doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(Document(page_content=content, metadata=metadata))

    # remove tiny/empty chunks
    filtered = [
        c
        for c in final_chunks
        if c.page_content and isinstance(c.page_content, str) and len(c.page_content.strip()) > 50
    ]
    print(f"[INGEST] {len(filtered)} chunks after filtering small/empty ones.")
    return filtered


# ---------- main ingest function ----------
def ingest_document(file_path: str, session_id: str):
    """
    Full ingestion pipeline for ONE PDF.
    Runs in FastAPI background task.
    """

    # 1. Parse
    md_text = pdf_to_markdown(file_path)
    docs = [
        Document(
            page_content=md_text,
            metadata={"source_file": os.path.basename(file_path)},
        )
    ]

    # 2. Chunk
    chunks = hierarchical_split(docs)
    if not chunks:
        raise ValueError("No valid chunks extracted from PDF.")

    texts = [c.page_content for c in chunks]

    # 3. Embeddings (batch)
    embeddings = get_embeddings()
    dense_vecs = embeddings.embed_documents(texts)
    print(f"[INGEST {session_id}] Embeddings computed for {len(texts)} chunks.")

    # 4. BM25 per session (in-memory)
    bm25 = BM25Encoder()
    bm25.fit(texts)
    _bm25_store[session_id] = bm25
    print(f"[INGEST {session_id}] BM25 fitted and stored in memory.")

    # 5. Prepare vectors
    vectors = []
    for i, chunk in enumerate(chunks):
        sparse_vec = bm25.encode_documents(chunk.page_content)
        metadata = chunk.metadata.copy()
        metadata["text"] = chunk.page_content
        vectors.append(
            {
                "id": str(uuid4()),
                "values": dense_vecs[i],
                "sparse_values": sparse_vec,
                "metadata": metadata,
            }
        )

    # 6. Upsert to Pinecone (namespace=session_id)
    index = get_index()
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=session_id)
        print(f"[INGEST {session_id}] Upserted batch {i // BATCH_SIZE + 1}")

    print(f"[INGEST {session_id}] Completed. {len(vectors)} vectors stored.")


def get_bm25_for_session(session_id: str) -> BM25Encoder:
    bm25 = _bm25_store.get(session_id)
    if bm25 is None:
        raise ValueError(f"BM25 not found in memory for session {session_id}")
    return bm25
