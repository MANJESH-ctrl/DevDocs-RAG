"""
Evaluation Configuration
========================
Central configuration for the RAG evaluation suite.
Sets up the evaluator LLM (Groq via OpenAI-compatible API),
evaluation parameters, and output paths.
"""

import os
import sys

# ── Make project root importable ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── API Keys (reuse existing project keys) ────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ── Evaluator LLM Config ─────────────────────────────────────
# RAGAS uses an LLM as a "judge" to score faithfulness, relevancy, etc.
# We reuse Groq (free tier) so there's zero additional cost.
EVALUATOR_MODEL = os.getenv("EVAL_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ── Evaluation Parameters ─────────────────────────────────────
NUM_TEST_QUESTIONS = int(os.getenv("NUM_TEST_QUESTIONS", "10"))
TOP_K = 10       # candidates fetched from retriever
FINAL_K = 5      # after reranking

# ── Paths ─────────────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TESTSET_DIR = os.path.join(EVAL_DIR, "testsets")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RAG_DOCS_DIR = os.path.join(PROJECT_ROOT, "RAG_docs")

os.makedirs(TESTSET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Helper: build the RAGAS evaluator LLM wrapper ─────────────
def get_evaluator_llm():
    """
    Returns a RAGAS-compatible LLM wrapper using Groq's OpenAI-compatible API.
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )
    return llm_factory(model=EVALUATOR_MODEL, client=client)


def get_evaluator_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)
