import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("INDEX_NAME")
CLOUD_REGION     = os.getenv("CLOUD_REGION", "us-west-1")

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
LLM_MODEL        = "llama-3.1-8b-instant"

BASE_DIR   = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
BATCH_SIZE    = 100
TOP_K         = 10
FINAL_K       = 5
