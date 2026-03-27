"""
End-to-End RAG Evaluation (No RAGAS — LLM-as-Judge via Groq)
=============================================================
Runs every test question through the live DevDocs-RAG pipeline
and evaluates with 4 metrics using Groq as the judge LLM.

Metrics:
  1. Faithfulness        — Is the answer grounded in the retrieved context?
  2. Context Precision   — Are the retrieved chunks relevant to the question?
  3. Context Recall      — Does the context cover the ground-truth answer?
  4. Answer Correctness  — Does the answer match the ground-truth answer?

All scores: 0.0 (worst) → 1.0 (best)

Usage:
    python -m evaluation.run_evaluation \
        --testset evaluation/testsets/testset_TensorFlow_<timestamp>.json \
        --session-id <id>
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

# ── Ensure project root is importable ─────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    EVALUATOR_MODEL,
    RESULTS_DIR,
)


# ═══════════════════════════════════════════════════════════════
#  STEP 1: Run questions through the RAG pipeline
# ═══════════════════════════════════════════════════════════════

def run_rag_pipeline(question: str, session_id: str) -> Dict:
    """Run one question through DevDocs-RAG. Returns answer + contexts."""
    from app.query import hybrid_retriever, format_docs, get_llm
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from app.query import TEMPLATE

    docs = hybrid_retriever(question, session_id)
    retrieved_contexts = [doc.page_content for doc in docs]

    context_str = format_docs(docs)
    prompt = PromptTemplate.from_template(TEMPLATE)
    chain = (
        {
            "context":  lambda _: context_str,
            "history":  lambda _: "No previous conversation.",
            "question": RunnablePassthrough(),
        }
        | prompt | get_llm() | StrOutputParser()
    )
    answer = chain.invoke(question)
    return {"response": answer, "retrieved_contexts": retrieved_contexts}


# ═══════════════════════════════════════════════════════════════
#  STEP 2: LLM-as-Judge scoring (pure Groq, no RAGAS)
# ═══════════════════════════════════════════════════════════════

def _ask_judge(prompt: str) -> float:
    """
    Send a scoring prompt to Groq and parse the float score out of the reply.
    Returns a float between 0.0 and 1.0.
    """
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content
    if raw is None:
        return 0.5  # fallback if content is None
    raw = raw.strip()
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))   # clamp to [0, 1]
    except ValueError:
        # Try to pull a number out of a longer reply
        import re
        match = re.search(r"\d+\.?\d*", raw)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.5  # fallback


def score_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """Is every claim in the answer supported by the context?"""
    ctx = "\n\n".join(contexts[:3])
    prompt = f"""You are an evaluation judge. Score whether the ANSWER is fully supported by the CONTEXT (no hallucinations).

QUESTION: {question}

CONTEXT:
{ctx}

ANSWER: {answer}

Score: 1.0 = every claim is supported | 0.0 = answer contains unsupported claims.
Reply with ONLY a decimal number between 0.0 and 1.0."""
    return _ask_judge(prompt)


def score_context_precision(question: str, contexts: List[str]) -> float:
    """Are the retrieved chunks relevant to the question?"""
    ctx = "\n\n---\n\n".join(contexts[:3])
    prompt = f"""You are an evaluation judge. Score how relevant the retrieved CONTEXT is to the QUESTION.

QUESTION: {question}

CONTEXT:
{ctx}

Score: 1.0 = all chunks are highly relevant | 0.0 = all chunks are irrelevant.
Reply with ONLY a decimal number between 0.0 and 1.0."""
    return _ask_judge(prompt)


def score_context_recall(contexts: List[str], reference: str) -> float:
    """Does the context contain all the information in the ground-truth answer?"""
    ctx = "\n\n".join(contexts[:3])
    prompt = f"""You are an evaluation judge. Score whether the CONTEXT contains all the information needed to produce the REFERENCE ANSWER.

CONTEXT:
{ctx}

REFERENCE ANSWER: {reference}

Score: 1.0 = context fully covers the reference answer | 0.0 = context is missing key information.
Reply with ONLY a decimal number between 0.0 and 1.0."""
    return _ask_judge(prompt)


def score_answer_correctness(answer: str, reference: str) -> float:
    """Does the answer factually match the ground-truth reference?"""
    prompt = f"""You are an evaluation judge. Score how factually correct the ANSWER is compared to the REFERENCE ANSWER.

REFERENCE ANSWER: {reference}

ANSWER: {answer}

Score: 1.0 = fully correct and complete | 0.0 = completely wrong or missing.
Reply with ONLY a decimal number between 0.0 and 1.0."""
    return _ask_judge(prompt)


def evaluate_sample(sample: Dict) -> Dict:
    """Score one sample across all 4 metrics."""
    q   = sample["user_input"]
    ans = sample["response"]
    ctx = sample["retrieved_contexts"]
    ref = sample.get("reference", sample.get("expected_output", ""))

    return {
        "faithfulness":       score_faithfulness(q, ans, ctx),
        "context_precision":  score_context_precision(q, ctx),
        "context_recall":     score_context_recall(ctx, ref),
        "answer_correctness": score_answer_correctness(ans, ref),
    }


# ═══════════════════════════════════════════════════════════════
#  STEP 3: Console output
# ═══════════════════════════════════════════════════════════════

def print_results(aggregate: Dict, num_samples: int):
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated : {num_samples}")
    print(f"  Evaluator model   : {EVALUATOR_MODEL}")
    print("-" * 60)

    for metric, score in aggregate.items():
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        if score >= 0.8:
            status = "✅"
        elif score >= 0.6:
            status = "🟡"
        elif score >= 0.4:
            status = "🟠"
        else:
            status = "🔴"
        print(f"  {status} {metric:<22s} {bar}  {score:.4f}")

    overall = sum(aggregate.values()) / len(aggregate)
    print("-" * 60)
    print(f"  🏆 Overall Score   {'':22s}  {overall:.4f}")
    print("=" * 60)

    # Quick diagnostics
    print("\n📋 Diagnostics:")
    if aggregate.get("faithfulness", 1) < 0.6:
        print("  ⚠  Low Faithfulness   → LLM is hallucinating. Improve the prompt or lower temperature.")
    if aggregate.get("context_precision", 1) < 0.6:
        print("  ⚠  Low Ctx Precision  → Irrelevant chunks retrieved. Raise score threshold in query.py.")
    if aggregate.get("context_recall", 1) < 0.6:
        print("  ⚠  Low Ctx Recall     → Missing info. Increase TOP_K or reduce CHUNK_SIZE.")
    if aggregate.get("answer_correctness", 1) < 0.6:
        print("  ⚠  Low Correctness    → Wrong answers. Fix retrieval first, then generation.")
    if all(v >= 0.8 for v in aggregate.values()):
        print("  ✅ All metrics excellent! Pipeline is performing well.")


# ═══════════════════════════════════════════════════════════════
#  STEP 4: Save reports
# ═══════════════════════════════════════════════════════════════

def save_reports(
    aggregate: Dict,
    per_question: List[Dict],
    eval_samples: List[Dict],
    testset_path: str,
    session_id: str,
) -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSON ──────────────────────────────────────────────────
    report = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "testset": testset_path,
            "session_id": session_id,
            "evaluator_model": EVALUATOR_MODEL,
            "num_samples": len(eval_samples),
            "method": "LLM-as-Judge (Groq, no RAGAS)",
        },
        "aggregate_scores": aggregate,
        "per_question_scores": per_question,
    }
    json_path = os.path.join(RESULTS_DIR, f"eval_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Markdown ──────────────────────────────────────────────
    overall = sum(aggregate.values()) / len(aggregate)
    lines = [
        "# 📊 DevDocs-RAG Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Test Set:** `{os.path.basename(testset_path)}`  ",
        f"**Session ID:** `{session_id}`  ",
        f"**Evaluator:** `{EVALUATOR_MODEL}` (LLM-as-Judge, no RAGAS)  ",
        f"**Samples:** {len(eval_samples)}",
        "",
        "---",
        "",
        "## 🎯 Aggregate Scores",
        "",
        "| Metric | Score | Rating |",
        "|--------|-------|--------|",
    ]
    for metric, score in aggregate.items():
        rating = "🟢 Excellent" if score >= 0.8 else "🟡 Good" if score >= 0.6 else "🟠 Fair" if score >= 0.4 else "🔴 Needs Work"
        lines.append(f"| **{metric}** | {score:.4f} | {rating} |")

    lines += [
        "",
        f"### 🏆 Overall Score: **{overall:.4f}**",
        "",
        "---",
        "",
        "## 📋 Per-Question Breakdown",
        "",
        "| # | Question | Faithfulness | Ctx Precision | Ctx Recall | Correctness |",
        "|---|----------|:---:|:---:|:---:|:---:|",
    ]
    for i, pq in enumerate(per_question):
        q = pq["question"][:55] + "..." if len(pq["question"]) > 55 else pq["question"]
        lines.append(
            f"| {i+1} | {q} "
            f"| {pq.get('faithfulness', 'N/A')} "
            f"| {pq.get('context_precision', 'N/A')} "
            f"| {pq.get('context_recall', 'N/A')} "
            f"| {pq.get('answer_correctness', 'N/A')} |"
        )

    md_path = os.path.join(RESULTS_DIR, f"eval_report_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation")
    parser.add_argument("--testset",    type=str, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 DevDocs-RAG — End-to-End Evaluation")
    print("=" * 60)

    # Load test set
    print(f"\n📂 Loading test set: {args.testset}")
    with open(args.testset, "r", encoding="utf-8") as f:
        testset = json.load(f)
    samples = testset["samples"]
    print(f"   Found {len(samples)} test samples")

    # Run RAG pipeline
    print(f"\n🔄 Running questions through the RAG pipeline...")
    eval_samples = []
    for i, sample in enumerate(samples):
        question = sample["user_input"]
        print(f"  [{i+1}/{len(samples)}] {question[:70]}...")
        try:
            result = run_rag_pipeline(question, args.session_id)
            eval_samples.append({
                "user_input":          question,
                "response":            result["response"],
                "retrieved_contexts":  result["retrieved_contexts"],
                "reference":           sample.get("expected_output", ""),
            })
        except Exception as e:
            print(f"    ⚠ Skipped: {e}")

    if not eval_samples:
        print("\n❌ No samples processed. Check your session ID.")
        return

    print(f"\n✅ Processed {len(eval_samples)}/{len(samples)} samples")

    # Score each sample
    print(f"\n⚙️  Scoring with LLM-as-Judge ({EVALUATOR_MODEL})...")
    per_question = []
    metrics_sum = {"faithfulness": 0, "context_precision": 0, "context_recall": 0, "answer_correctness": 0}

    for i, sample in enumerate(eval_samples):
        print(f"  [{i+1}/{len(eval_samples)}] Scoring...", end="\r")
        scores = evaluate_sample(sample)
        per_question.append({"question": sample["user_input"], **{k: round(v, 4) for k, v in scores.items()}})
        for k, v in scores.items():
            metrics_sum[k] += v

    n = len(eval_samples)
    aggregate = {k: round(v / n, 4) for k, v in metrics_sum.items()}

    # Output
    print_results(aggregate, n)
    json_path, md_path = save_reports(aggregate, per_question, eval_samples, args.testset, args.session_id)
    print(f"\n📄 JSON  → {json_path}")
    print(f"📝 Report→ {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()