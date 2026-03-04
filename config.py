import os
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import tiktoken

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("INDEX_NAME")
CLOUD_REGION     = os.getenv("CLOUD_REGION", "us-east-1")
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
LLM_MODEL        = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
BATCH_SIZE    = 200
TOP_K         = 10
FINAL_K       = 5

# ── Module-level singletons — built once, reused always ──────
HEADERS = [
    ("#",     "Header 1"),
    ("##",    "Header 2"),
    ("###",   "Header 3"),
    ("####",  "Header 4"),
    ("#####", "Header 5"),
]

MD_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS,
    strip_headers=False,
)

ENCODING = tiktoken.get_encoding("cl100k_base")

TOKEN_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", " "],
)
