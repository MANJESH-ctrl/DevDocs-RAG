# 📊 DevDocs-RAG Evaluation Suite

A production-grade RAG evaluation system built on **RAGAS** (Retrieval-Augmented Generation Assessment) — the industry-standard framework for evaluating RAG pipelines.

## 🎯 What Gets Evaluated

### 6 Metrics Across 3 Categories

| Category | Metric | What It Measures |
|----------|--------|-----------------|
| **Retrieval** | Context Precision | Are retrieved chunks relevant and well-ranked? |
| **Retrieval** | Context Recall | Did retrieval capture all needed information? |
| **Generation** | Faithfulness | Is the answer grounded in context? (hallucination detection) |
| **Generation** | Answer Relevancy | Does the response directly address the question? |
| **End-to-End** | Answer Correctness | Factual accuracy compared to ground truth |
| **End-to-End** | Answer Similarity | Semantic closeness to reference answer |

> All scores range from **0.0** (worst) to **1.0** (best).

---

## 🚀 Setup

### Install Dependencies

```bash
pip install -r evaluation/requirements.txt
```

> The evaluation uses your existing **Groq API key** as the evaluator LLM — zero additional cost.

---

## 📋 Workflow (3 Steps)

### Step 1: Generate Test Set from Your PDFs

```bash
# Generate QA pairs from all PDFs in RAG_docs/
python -m evaluation.generate_testset

# Or from a specific PDF
python -m evaluation.generate_testset --pdf langchain_rag_full_tutorial.pdf

# Control number of questions
python -m evaluation.generate_testset --num-questions 20 --questions-per-chunk 3
```

**Output:** `evaluation/testsets/testset_<name>_<timestamp>.json`

Each QA pair includes:
- `user_input` — the question
- `expected_output` — reference answer (ground truth)
- `reference_contexts` — source chunks
- `metadata` — question type (SIMPLE/REASONING/SPECIFIC), source file, headers

### Step 2: Ingest a Document (if not already done)

Upload a PDF through the web UI or API to get a `session_id`:
```bash
curl -X POST http://localhost:8000/upload -F "file=@RAG_docs/langchain_rag_full_tutorial.pdf"
# Response: {"session_id": "abc-123", "status": "processing"}
```

### Step 3: Run Evaluation

#### Full End-to-End Evaluation (6 metrics)
```bash
python -m evaluation.run_evaluation \
  --testset evaluation/testsets/testset_all_20250322.json \
  --session-id abc-123
```

#### Retrieval-Only Evaluation (2 metrics)
```bash
python -m evaluation.evaluate_retrieval \
  --testset evaluation/testsets/testset_all_20250322.json \
  --session-id abc-123
```

---

## 📊 Output

### Console
```
════════════════════════════════════════════════════════════
📊 EVALUATION RESULTS
════════════════════════════════════════════════════════════
  Samples evaluated: 10
────────────────────────────────────────────────────────────
  ✅ faithfulness                █████████████████████████░░░░░ 0.8500
  ✅ answer_relevancy            ████████████████████████░░░░░░ 0.8200
  🟡 context_precision           ██████████████████░░░░░░░░░░░░ 0.6100
  ✅ context_recall              ██████████████████████████░░░░ 0.8700
  ✅ factual_correctness         ████████████████████████░░░░░░ 0.8000
  ✅ semantic_similarity         █████████████████████████████░ 0.9500
════════════════════════════════════════════════════════════
```

### Files
- `evaluation/results/eval_results_<timestamp>.json` — full per-question scores
- `evaluation/results/eval_report_<timestamp>.md` — formatted markdown report

---

## 🔧 How It Works Under the Hood

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                       │
│                                                              │
│  Test Set (JSON)                                             │
│       ↓                                                      │
│  For each question:                                          │
│    1. hybrid_retriever() → retrieved contexts                │
│    2. LLM chain → generated answer                           │
│       ↓                                                      │
│  Collect: question, answer, contexts, ground_truth           │
│       ↓                                                      │
│  RAGAS Evaluation (LLM-as-Judge via Groq)                   │
│    • Faithfulness: verify claims against context             │
│    • Relevancy: check answer addresses the question          │
│    • Precision: judge chunk relevance & ranking              │
│    • Recall: check if context covers ground truth            │
│    • Correctness: compare with reference answer              │
│    • Similarity: semantic embedding distance                 │
│       ↓                                                      │
│  JSON + Markdown Reports                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Interpreting Results

| Score Range | Rating | Action |
|-------------|--------|--------|
| **0.8 – 1.0** | 🟢 Excellent | Pipeline is performing well |
| **0.6 – 0.8** | 🟡 Good | Minor improvements possible |
| **0.4 – 0.6** | 🟠 Fair | Investigate and tune |
| **< 0.4** | 🔴 Needs Work | Significant issues to address |

### Diagnostic Cheat Sheet

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Low Faithfulness | LLM hallucinating | Improve prompt, lower temperature, or use a stronger model |
| Low Answer Relevancy | Off-topic answers | Improve prompt template, check context quality |
| Low Context Precision | Irrelevant chunks retrieved | Increase score threshold (>0.2), tune FINAL_K |
| Low Context Recall | Missing information | Increase TOP_K, reduce chunk size, check chunking strategy |
| Low Answer Correctness | Wrong facts in answers | Usually a combination — fix retrieval first, then generation |

---

## 📁 File Structure

```
evaluation/
├── __init__.py                # Package init
├── config.py                  # Evaluator LLM setup & paths
├── generate_testset.py        # Synthetic QA generator from PDFs
├── run_evaluation.py          # Main end-to-end evaluation (6 metrics)
├── evaluate_retrieval.py      # Retrieval-only evaluation (2 metrics)
├── requirements.txt           # Additional pip dependencies
├── README.md                  # This file
├── testsets/                  # Generated test sets (JSON)
│   └── testset_*.json
└── results/                   # Evaluation results
    ├── eval_results_*.json
    ├── eval_report_*.md
    └── retrieval_eval_*.json
```
