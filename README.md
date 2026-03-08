<div align="center">

# 🧠 DevDocs RAG

### Ask anything about your developer documentation. Get precise, cited answers — instantly.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge)](https://retardyadav-dev-docs-rag-app.hf.space/)
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
│                        INGESTION PIPELINE                       │
│                                                                 │
│  PDF Upload → pymupdf4llm → Semantic Chunking → BGE Embeddings  │
│                                    ↓              ↓             | 
│                              BM25 Index     Pinecone Vector DB  |
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                          │
│                                                                 │
│  User Question → BGE Embeddings                                 │
│                       ↓                                         │
│             ┌─── Dense Search (Pinecone) ───┐                   │
│             └─── Sparse Search (BM25)    ───┘                   │
│                       ↓                                         │
│              Reciprocal Rank Fusion                             │
│                       ↓                                         │
│           CrossEncoder Neural Reranking                         │
│                       ↓                                         │
│         Top-K Chunks → Groq LLM (LLaMA 3.1)                     │
│                       ↓                                         │
│            SSE Streaming → Browser                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | FastAPI + Uvicorn | Async REST API & SSE streaming |
| **PDF Parsing** | pymupdf4llm | Markdown-aware PDF extraction |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Dense semantic vector representations |
| **Sparse Retrieval** | BM25 (rank_bm25) | Keyword-aware lexical search |
| **Vector Database** | Pinecone | Scalable ANN vector search |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural passage reranking |
| **LLM** | Groq · LLaMA 3.1 8B Instant | Ultra-fast answer generation |
| **Streaming** | Server-Sent Events (SSE) | Real-time token streaming |
| **NLP Utilities** | NLTK | Tokenization & text preprocessing |
| **Frontend** | Vanilla JS + CSS | Zero-dependency chat UI |
| **Containerization** | Docker (multi-stage build) | Reproducible deployment |
| **Deployment** | HuggingFace Spaces | Free cloud hosting with 16GB RAM |

---

## 🔬 RAG Pipeline — Deep Dive

### 1. Ingestion
- PDF is parsed using **pymupdf4llm** which preserves markdown structure (headers, lists, tables) rather than raw text dumps
- Document is split into semantically meaningful chunks with configurable overlap
- Each chunk is embedded using **BAAI/bge-small-en-v1.5** — a top-ranked MTEB model optimized for retrieval
- Embeddings stored in **Pinecone** with metadata (source, section, page)
- Parallel **BM25 index** built for sparse keyword retrieval and persisted to disk

### 2. Hybrid Retrieval
- Query is embedded with the same BGE model for dense search
- Simultaneously, BM25 scores the query against all indexed chunks
- Results from both retrievers are merged using **Reciprocal Rank Fusion (RRF)** — a parameter-free rank merging algorithm that consistently outperforms score-based fusion

### 3. Neural Reranking
- Fused candidates are passed through **cross-encoder/ms-marco-MiniLM-L-6-v2**
- Unlike bi-encoders, CrossEncoders perform full attention across query+passage pairs for precise relevance scoring
- Top-K results selected after reranking for final context assembly

### 4. Generation & Streaming
- Curated context + conversation history sent to **Groq's LLaMA 3.1 8B Instant**
- Response streamed token-by-token via **SSE** to the browser
- Frontend renders markdown in real-time as tokens arrive

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Pinecone API key (free tier)
- Groq API key (free tier)

### Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/DevDocs-RAG.git
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
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `POST /upload` | POST | Upload PDF, returns `session_id` |
| `GET /status/{session_id}` | GET | Poll ingestion progress |
| `POST /chat/{session_id}` | POST | Send question, returns SSE stream |

### Chat Request
```json
{
  "question": "What are the key parameters of numpy.einsum?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

### SSE Stream Events
```
data: {"type": "sources", "sources": [...]}
data: {"type": "token",   "content": "The "}
data: {"type": "token",   "content": "einsum "}
data: {"type": "done"}
```

---

## 🐳 Docker — Multi-Stage Build

The Dockerfile uses a **two-stage build** to keep the runtime image lean:

- **Builder stage** — installs `build-essential`, compiles all packages, downloads and warms up both ML models
- **Runtime stage** — copies only the venv and cached models, no build tools, minimal attack surface

Models are **baked into the image at build time** via `warmup_models.py` — zero cold-start latency on first request.

---

## 📁 Project Structure

```
DevDocs-RAG/
├── main.py              # FastAPI app, routes, SSE streaming
├── ingestion.py         # PDF parsing, chunking, embedding, indexing
├── query.py             # Hybrid retrieval, reranking, LLM generation
├── config.py            # Environment config & model initialization
├── warmup_models.py     # Pre-loads models at Docker build time
├── static/
│   └── index.html       # Full chat UI (zero dependencies)
├── Dockerfile           # Multi-stage Docker build
├── requirements.in      # Python dependencies
└── README.md
```

---

## 🌐 Deployment

Deployed on **HuggingFace Spaces** (Docker SDK):
- ✅ 16GB RAM free tier — enough for both ML models with headroom
- ✅ Public spaces never sleep — no cold starts
- ✅ HuggingFace model hub on same network — instant model downloads at build time
- ✅ Zero cost, zero credit card

**Live:** https://huggingface.co/spaces/retardyadav/dev-docs-rag-app

---

## 🧪 Tested Documents

| Document | Pages | Result |
|---|---|---|
| NumPy User Guide | 500+ | ✅ |
| MLflow Documentation | 200+ | ✅ |
| TensorFlow User Guide | 300+ | ✅ |
| spaCy Tutorial | 150+ | ✅ |
| LightGBM Docs | 100+ | ✅ |
| LangChain + Ollama Guide | 80+ | ✅ |

---

## 🔮 Roadmap

- [ ] Multi-document sessions (query across multiple PDFs simultaneously)
- [ ] URL ingestion (scrape and index documentation websites)
- [ ] Persistent sessions with database backend
- [ ] Syntax highlighting via highlight.js
- [ ] Export conversation as PDF/Markdown

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using FastAPI · Pinecone · Groq · HuggingFace · CrossEncoder · Docker

**If this project helped you, consider starring ⭐ the repo**

</div>
