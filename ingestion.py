import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4

# === CONFIG ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")                              
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")               
CLOUD_REGION = os.getenv("CLOUD_REGION")    
FOLDER = os.getenv("FOLDER")  


def pdf_to_markdown(filepath):
    """Convert PDF to Markdown using unstructured"""
    print(f"Converting: {os.path.basename(filepath)}")
    elements = partition_pdf(
        filename=filepath,
        strategy="auto",
        languages=["eng"],
    )
    md_text = "\n\n".join([str(el) for el in elements if el.text.strip()])
    return md_text.strip()

def load_pdfs_as_markdown(folder_path):
    """Load all PDFs and convert to Markdown Documents"""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            md_content = pdf_to_markdown(filepath)
            if md_content:  # skip empty files
                doc = Document(
                    page_content=md_content,
                    metadata={"source_file": filename}
                )
                documents.append(doc)
    return documents

def hierarchical_split(documents, chunk_size=450, chunk_overlap=100):
    """Hierarchical splitting: Headers first → Token-based split for large chunks"""
    # Markdown Header Splitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # Token-based splitter for large sections
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )

    final_chunks = []

    for doc in documents:
        # Step 1: Split by headers
        md_chunks = md_splitter.split_text(doc.page_content)

        for md_chunk in md_chunks:
            content = md_chunk.page_content.strip()
            metadata = {**doc.metadata, **md_chunk.metadata}

            # Step 2: If chunk is too big, split further with token splitter
            if len(content) > 2800:
                temp_doc = Document(page_content=content, metadata=metadata)
                sub_chunks = token_splitter.split_documents([temp_doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(Document(page_content=content, metadata=metadata))

    # Optional: Filter very tiny chunks
    final_chunks = [c for c in final_chunks if len(c.page_content.strip()) >= 120]

    return final_chunks


# main

print("Starting processing...")
docs = load_pdfs_as_markdown(FOLDER)
print(f"Loaded {len(docs)} documents\n")

chunks = hierarchical_split(docs, chunk_size=450, chunk_overlap=100)
print(f"Created {len(chunks)} chunks\n")


# === Initialize Pinecone client ===
pc = Pinecone(api_key=PINECONE_API_KEY)

# === Create index if it doesn't exist ===
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,                         
        metric="cosine",                   
        spec=ServerlessSpec(
            cloud="aws",
            region=CLOUD_REGION
        )
    )
    print("Index created (may take sometime to be ready)")
else:
    print(f"Using existing index: {INDEX_NAME}")


# === Setup Embedding Model ===
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},                    
    encode_kwargs={'normalize_embeddings': True}
)

# === Create / Connect to Vector Store & Upsert Chunks ===
print(f"Storing {len(chunks)} chunks into Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    # namespace="dev_docs"   
)

print(f"Successfully stored {len(chunks)} chunks in Pinecone index '{INDEX_NAME}'!")

# === Test Retrieval ===
print("\n Quick test retrieval...")
query = "How do I create a zero-shot agent in LangChain with Ollama?"
results = vectorstore.similarity_search(query, k=4)

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {doc.metadata.get('source_file', 'unknown')}")
    print(f"Header path: {doc.metadata.get('Header 1', '')} > {doc.metadata.get('Header 2', '')}")
    print(f"Content preview: {doc.page_content[:300]}...")