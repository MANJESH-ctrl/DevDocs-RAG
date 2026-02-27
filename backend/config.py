# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("INDEX_NAME")
CLOUD_REGION     = os.getenv("CLOUD_REGION", "us-west-1")

# --- Embedding ---
EMBEDDING_MODEL  = os.getenv(
    "EMBEDDING_MODEL",
    # "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Groq ---
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
LLM_MODEL        = "llama-3.1-8b-instant"

# --- Paths ---
BASE_DIR  = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Chunking / retrieval params ---
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
BATCH_SIZE    = 100   # Pinecone upsert batch size
TOP_K         = 10    # candidates from Pinecone
FINAL_K       = 5     # final chunks sent to LLM
