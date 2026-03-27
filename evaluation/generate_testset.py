"""
Synthetic Test Set Generator
=============================
Generates QA pairs from PDF documents in RAG_docs/ for evaluation.

Uses the Groq LLM to create diverse question-answer pairs from document
chunks, producing a structured test set with:
  - user_input:          the question
  - expected_output:     the reference answer (ground truth)
  - reference_contexts:  the source chunks the answer was derived from

Usage:
    python -m evaluation.generate_testset                           # all PDFs
    python -m evaluation.generate_testset --pdf langchain_rag_full_tutorial.pdf
    python -m evaluation.generate_testset --num-questions 20
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict

# ── Ensure project root is importable ─────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import MD_SPLITTER, ENCODING, TOKEN_SPLITTER, CHUNK_SIZE
from app.ingestion import pdf_to_markdown, hierarchical_split
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from evaluation.config import (
    GROQ_API_KEY,
    EVALUATOR_MODEL,
    TESTSET_DIR,
    RAG_DOCS_DIR,
    NUM_TEST_QUESTIONS,
)

# ── Prompt for QA generation ─────────────────────────────────
QA_GENERATION_PROMPT = """You are an expert at creating evaluation questions for a Retrieval-Augmented Generation (RAG) system.

Given the following document chunk from developer documentation, generate {num_questions} diverse question-answer pairs.

Requirements:
1. Questions should be varied in type:
   - SIMPLE: Direct factual questions answerable from the chunk
   - REASONING: Questions requiring understanding/inference from the chunk
   - SPECIFIC: Questions about specific code, APIs, parameters, or configurations
2. Answers must be ONLY based on the provided chunk — do not use external knowledge
3. Answers should be detailed and helpful (2-4 sentences minimum)
4. Questions should be the kind a developer would actually ask

DOCUMENT CHUNK:
---
{chunk_text}
---

Respond in valid JSON format ONLY — no markdown, no commentary:
[
  {{
    "question": "...",
    "answer": "...",
    "type": "SIMPLE|REASONING|SPECIFIC"
  }}
]"""


def generate_qa_from_chunk(
    llm: ChatGroq,
    chunk_text: str,
    num_questions: int = 3,
    max_retries: int = 3,
) -> List[Dict]:
    """Generate QA pairs from a single chunk using the Groq LLM."""
    prompt = QA_GENERATION_PROMPT.format(
        num_questions=num_questions,
        chunk_text=chunk_text[:3000],  # cap to avoid token limits
    )

    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[: content.rfind("```")]
                content = content.strip()

            qa_pairs = json.loads(content)
            if isinstance(qa_pairs, list) and len(qa_pairs) > 0:
                return qa_pairs

        except json.JSONDecodeError:
            print(f"    ⚠ JSON parse error on attempt {attempt + 1}, retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"    ⚠ Error on attempt {attempt + 1}: {e}")
            time.sleep(2)

    return []


def process_pdf(
    pdf_path: str,
    llm: ChatGroq,
    num_questions: int = NUM_TEST_QUESTIONS,
    questions_per_chunk: int = 3,
) -> List[Dict]:
    """
    Process a single PDF: parse → chunk → generate QA pairs.
    Returns a list of evaluation samples.
    """
    pdf_name = os.path.basename(pdf_path)
    print(f"\n📄 Processing: {pdf_name}")

    # 1. Parse PDF to markdown
    print("  ├─ Parsing PDF to markdown...")
    md_text = pdf_to_markdown(pdf_path)
    docs = [Document(page_content=md_text, metadata={"source_file": pdf_name})]

    # 2. Chunk the document
    print("  ├─ Chunking document...")
    chunks = hierarchical_split(docs)
    print(f"  ├─ Got {len(chunks)} chunks")

    if not chunks:
        print("  └─ ❌ No valid chunks extracted, skipping.")
        return []

    # 3. Select diverse chunks for QA generation
    # Pick evenly spaced chunks to cover the full document
    total_needed = max(1, num_questions // questions_per_chunk)
    step = max(1, len(chunks) // total_needed)
    selected_indices = list(range(0, len(chunks), step))[:total_needed]

    print(f"  ├─ Generating QA pairs from {len(selected_indices)} chunks...")

    samples = []
    for idx in selected_indices:
        chunk = chunks[idx]
        qa_pairs = generate_qa_from_chunk(
            llm, chunk.page_content, questions_per_chunk
        )

        for qa in qa_pairs:
            sample = {
                "user_input": qa.get("question", ""),
                "expected_output": qa.get("answer", ""),
                "reference_contexts": [chunk.page_content],
                "metadata": {
                    "source_file": pdf_name,
                    "question_type": qa.get("type", "SIMPLE"),
                    "chunk_index": idx,
                    "headers": {
                        k: v
                        for k, v in chunk.metadata.items()
                        if k.startswith("Header")
                    },
                },
            }
            samples.append(sample)

        print(f"  │  ├─ Chunk {idx}: generated {len(qa_pairs)} QA pairs")

        # Rate limiting for Groq free tier
        time.sleep(1)

    print(f"  └─ ✅ Total QA pairs: {len(samples)}")
    return samples


def save_testset(samples: List[Dict], filename: str) -> str:
    """Save test set to JSON file and return the path."""
    filepath = os.path.join(TESTSET_DIR, filename)
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "generator": "DevDocs-RAG Evaluation Suite",
            "evaluator_model": EVALUATOR_MODEL,
        },
        "samples": samples,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test set for RAG evaluation")
    parser.add_argument("--pdf", type=str, default=None, help="Specific PDF filename in RAG_docs/ (default: all PDFs)")
    parser.add_argument("--num-questions", type=int, default=NUM_TEST_QUESTIONS, help=f"Total questions to generate per PDF (default: {NUM_TEST_QUESTIONS})")
    parser.add_argument("--questions-per-chunk", type=int, default=3, help="Questions per chunk (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 DevDocs-RAG — Synthetic Test Set Generator")
    print("=" * 60)

    # ── Init LLM ──────────────────────────────────────────────
    llm = ChatGroq(
        model=EVALUATOR_MODEL,
        temperature=0.3,  # slight creativity for diverse questions
        groq_api_key=GROQ_API_KEY,
    )

    # ── Select PDFs ───────────────────────────────────────────
    if args.pdf:
        pdf_files = [args.pdf]
    else:
        pdf_files = [f for f in os.listdir(RAG_DOCS_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found in RAG_docs/")
        return

    print(f"\n📂 PDFs to process: {len(pdf_files)}")
    for f in pdf_files:
        print(f"  • {f}")

    # ── Process each PDF ──────────────────────────────────────
    all_samples = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAG_DOCS_DIR, pdf_file)
        if not os.path.exists(pdf_path):
            print(f"\n⚠ File not found: {pdf_path}, skipping.")
            continue

        samples = process_pdf(
            pdf_path, llm, args.num_questions, args.questions_per_chunk
        )
        all_samples.extend(samples)

    if not all_samples:
        print("\n❌ No QA pairs generated. Check your PDFs and API key.")
        return

    # ── Save test set ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.pdf:
        base_name = os.path.splitext(args.pdf)[0]
        filename = f"testset_{base_name}_{timestamp}.json"
    else:
        filename = f"testset_all_{timestamp}.json"

    filepath = save_testset(all_samples, filename)

    print(f"\n{'=' * 60}")
    print(f"✅ Test set saved to: {filepath}")
    print(f"   Total QA pairs: {len(all_samples)}")
    print(f"   Question types:")
    types = {}
    for s in all_samples:
        t = s["metadata"]["question_type"]
        types[t] = types.get(t, 0) + 1
    for t, count in sorted(types.items()):
        print(f"     • {t}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
