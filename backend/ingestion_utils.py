import os
from uuid import uuid4
from typing import List

# from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
# from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from pinecone_text.sparse import BM25Encoder
# import tiktoken

from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    CLOUD_REGION,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)

_embeddings = None       # lazy
_pc_index   = None       # lazy
_bm25_store = {}         # session_id -> BM25Encoder


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_index():
    global _pc_index
    if _pc_index is None:
        from pinecone import Pinecone, ServerlessSpec
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


def pdf_to_markdown(filepath: str) -> str:
    from unstructured.partition.pdf import partition_pdf  
    elements = partition_pdf(
        filename=filepath,
        strategy="fast",
        infer_table_structure=False,
        languages=["eng"],
    )
    md_text = ""
    for el in elements:
        if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
            md_text += f"\n{el.metadata.text_as_html}"
        elif el.text:
            md_text += f"\n{el.text}"
    return md_text.strip()


def hierarchical_split(documents: List[Document]) -> List[Document]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    import tiktoken
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
        for md_chunk in md_splitter.split_text(doc.page_content):
            content = (md_chunk.page_content or "").strip()
            if not content:
                continue
            metadata = {**doc.metadata, **md_chunk.metadata}
            tokens = encoding.encode(content)
            if len(tokens) > CHUNK_SIZE:
                sub_docs = token_splitter.split_documents(
                    [Document(page_content=content, metadata=metadata)]
                )
                final_chunks.extend(sub_docs)
            else:
                final_chunks.append(Document(page_content=content, metadata=metadata))

    return [
        c
        for c in final_chunks
        if c.page_content and isinstance(c.page_content, str) and len(c.page_content.strip()) > 50
    ]


def ingest_document(file_path: str, session_id: str):
    from pinecone_text.sparse import BM25Encoder
    from config import UPLOAD_DIR  # local import to avoid cycles

    md_text = pdf_to_markdown(file_path)
    docs = [Document(page_content=md_text, metadata={"source_file": os.path.basename(file_path)})]

    chunks = hierarchical_split(docs)
    if not chunks:
        raise ValueError("No valid chunks extracted from PDF.")

    texts = [c.page_content for c in chunks]

    embeddings = get_embeddings()
    dense_vecs = embeddings.embed_documents(texts)

    bm25 = BM25Encoder()
    bm25.fit(texts)
    _bm25_store[session_id] = bm25

    index = get_index()
    vectors = []
    for i, chunk in enumerate(chunks):
        sparse_vec = bm25.encode_documents(chunk.page_content)
        meta = chunk.metadata.copy()
        meta["text"] = chunk.page_content
        vectors.append(
            {"id": str(uuid4()), "values": dense_vecs[i], "sparse_values": sparse_vec, "metadata": meta}
        )

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=session_id)


def get_bm25_for_session(session_id: str):
    if session_id not in _bm25_store:
        raise ValueError(f"BM25 not found for session {session_id}")
    return _bm25_store[session_id]
