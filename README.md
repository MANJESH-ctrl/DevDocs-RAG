---
title: Dev Docs Rag App
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">

# 🧠 DevDocs RAG

### Ask anything about your developer documentation. Get precise, cited answers — instantly.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/retardyadav/dev-docs-rag-app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🚩 The Problem

Developer documentation is massive, scattered, and hard to navigate. Whether it's a 300-page NumPy guide, a TensorFlow user manual, or an internal API spec — finding the **exact answer** to a specific question means endless Ctrl+F searches, tab-switching, and reading paragraphs that don't answer your question.

**DevDocs RAG solves this.** Drop any PDF developer document, ask your question in plain English, and get a precise, cited answer in seconds — powered by a production-grade Retrieval-Augmented Generation pipeline.

---

## ✨ Key Features

- 📄 **Upload any PDF** — developer docs, API references, research papers, user manuals
- 🔍 **Hybrid Search** — Dense vector search + BM25 sparse retrieval combined for superior recall
- 🎯 **Neural Reranking** — CrossEncoder re-scores retrieved chunks for maximum precision
- ⚡ **Streaming Responses** — Token-by-token SSE streaming, answers appear as they generate
- 📎 **Source Citations** — Every answer cites the exact document sections it used
- 🧠 **Conversation Memory** — Multi-turn chat with 6-message rolling context window
- 🎨 **Rich Markdown Rendering** — Headers, bold, lists, code blocks with syntax highlighting & copy button
- 🐳 **Fully Dockerized** — Single `docker build` + `docker run`, works anywhere

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  PDF Upload → pymupdf4llm → Semantic Chunking → BGE Embeddings  │
│                                    ↓              ↓             │
│                              BM25 Index     Pinecone Vector DB   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                  │
│  User Question → BGE Embeddings                                  │
│                       ↓                                          │
│             ┌─── Dense Search (Pinecone) ───┐                    │
│             └─── Sparse Search (BM25)    ───┘                    │
│                       ↓                                          │
│              Hybrid Score Fusion                                 │
│                       ↓                                          │
│           CrossEncoder Neural Reranking                          │
│                       ↓                                          │
│         Top-K Chunks → Groq LLM (LLaMA 3.1)                    │
│                       ↓                                          │
│            SSE Streaming → Browser                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | FastAPI + Uvicorn | Async REST API & SSE streaming |
| **PDF Parsing** | pymupdf4llm | Markdown-aware PDF extraction |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Dense semantic vector representations |
| **Sparse Retrieval** | BM25 (pinecone-text) | Keyword-aware lexical search |
| **Vector Database** | Pinecone | Scalable ANN vector search |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural passage reranking |
| **LLM** | Groq · LLaMA 3.1 8B Instant | Ultra-fast answer generation |
| **Streaming** | Server-Sent Events (SSE) | Real-time token streaming |
| **Frontend** | Vanilla JS + CSS | Zero-dependency chat UI |
| **Containerization** | Docker (multi-stage build) | Reproducible deployment |
| **Deployment** | HuggingFace Spaces | Free cloud hosting with 16GB RAM |

---

## 📊 Evaluation Results

The pipeline was evaluated using a **custom LLM-as-Judge framework** — prompt-engineered scoring rubrics powered by Groq (LLaMA 3.1 8B), evaluating 4 core RAG metrics on a synthetic test set generated from the TensorFlow User Guide.

> No off-the-shelf eval library was used — the evaluator was built from scratch for stability and control.

### Scores (TensorFlow User Guide · 9 samples)

| Metric | Score | Rating |
|--------|-------|--------|
| 🔍 **Faithfulness** | 0.82 | ✅ Excellent |
| 🎯 **Context Precision** | 0.59 | 🟠 Fair |
| 📚 **Context Recall** | 0.80 | ✅ Excellent |
| ✅ **Answer Correctness** | 0.81 | ✅ Excellent |
| 🏆 **Overall** | **0.75** | 🟡 Good |

### What Each Metric Means

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (hallucination detection) |
| **Context Precision** | Are retrieved chunks relevant to the question? |
| **Context Recall** | Does the context cover all information needed? |
| **Answer Correctness** | Does the answer match the ground-truth reference? |

### Key Takeaways

- **Faithfulness 0.82** — The LLM rarely hallucinates; answers stay grounded in the document
- **Context Recall 0.80** — Hybrid retrieval (dense + BM25) successfully captures most relevant chunks
- **Answer Correctness 0.81** — Answers are factually accurate against ground truth
- **Context Precision 0.59** — Some loosely related chunks slip through; tuning the score threshold is the next improvement

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Pinecone API key (free tier)
- Groq API key (free tier)

### Run Locally

```bash
# Clone
git clone https://github.com/MANJESH-ctrl/DevDocs-RAG.git
cd DevDocs-RAG

# Configure environment
cp .env.example .env
# Edit .env with your keys

# Build & run
docker build -t devdocs-rag .
docker run -p 8000:8000 --env-file .env devdocs-rag
```

Open `http://localhost:8000` — done.

### Environment Variables

```env
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_index_name
CLOUD_REGION=us-east-1
GROQ_API_KEY=your_groq_key
LLM_MODEL=llama-3.1-8b-instant
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `POST /upload` | POST | Upload PDF, returns `session_id` |
| `GET /status/{session_id}` | GET | Poll ingestion progress |
| `POST /chat/{session_id}` | POST | Send question, returns SSE stream |

---

## 📁 Project Structure

```
DevDocs-RAG/
├── main.py              # FastAPI app, routes, SSE streaming
├── app/
│   ├── config.py        # Environment config & model initialization
│   ├── ingestion.py     # PDF parsing, chunking, embedding, indexing
│   ├── query.py         # Hybrid retrieval, reranking, LLM generation
│   └── warmup_models.py # Pre-loads models at Docker build time
├── evaluation/
│   ├── config.py        # Evaluator config
│   ├── run_evaluation.py        # End-to-end evaluation (4 metrics)
│   ├── evaluate_retrieval.py    # Retrieval-only evaluation
│   ├── generate_testset.py      # Synthetic QA generator from PDFs
│   └── results/         # Saved evaluation reports
├── static/
│   └── index.html       # Full chat UI (zero dependencies)
├── RAG_docs/            # Place your PDFs here
├── Dockerfile           # Multi-stage Docker build
├── requirements.in      # Python dependencies
└── .env.example         # Environment variable template
```

---

## 🌐 Deployment

Deployed on **HuggingFace Spaces** (Docker SDK):
- ✅ 16GB RAM free tier — enough for both ML models with headroom
- ✅ Public spaces never sleep — no cold starts
- ✅ Zero cost, zero credit card

**Live:** https://huggingface.co/spaces/retardyadav/dev-docs-rag-app

---

## 🧪 Tested Documents

| Document | Pages | Result |
|---|---|---|
| TensorFlow User Guide | 300+ | ✅ |
| NumPy User Guide | 500+ | ✅ |
| MLflow Documentation | 200+ | ✅ |
| spaCy Tutorial | 150+ | ✅ |
| LightGBM Docs | 100+ | ✅ |
| LangChain + Ollama Guide | 80+ | ✅ |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using FastAPI · Pinecone · Groq · HuggingFace · CrossEncoder · Docker

**If this project helped you, consider starring ⭐ the repo**

</div>
