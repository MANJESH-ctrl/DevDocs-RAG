"""
Isolated Retrieval Evaluation
==============================
Tests ONLY the retrieval component (hybrid_retriever) without LLM generation.
Useful for diagnosing whether issues are in retrieval or generation,
and for A/B testing retrieval parameter changes (TOP_K, FINAL_K, score threshold).

Metrics:
  - Context Precision   — Are retrieved chunks relevant and well-ranked?
  - Context Recall      — Did retrieval capture all necessary information?

Usage:
    python -m evaluation.evaluate_retrieval --testset evaluation/testsets/testset_all_20250322.json --session-id <id>
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

# ── Ensure project root is importable ─────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.config import (
    RESULTS_DIR,
    EVALUATOR_MODEL,
    get_evaluator_llm,
)


def retrieve_contexts(question: str, session_id: str) -> List[str]:
    """
    Run only the retrieval step (no LLM generation).
    Returns the text content of retrieved document chunks.
    """
    from app.query import hybrid_retriever
    docs = hybrid_retriever(question, session_id)
    return [doc.page_content for doc in docs]


def evaluate_retrieval_metrics(eval_samples: List[Dict]):
    """
    Evaluate retrieval quality using RAGAS context-specific metrics.
    """
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics.collections import (
        ContextPrecisionWithoutReference,
        ContextRecall,
)

    ragas_samples = []
    for s in eval_samples:
        sample = SingleTurnSample(
            user_input=s["user_input"],
            response=s.get("expected_output", ""),  # use expected as proxy
            retrieved_contexts=s["retrieved_contexts"],
            reference=s.get("expected_output", ""),
        )
        ragas_samples.append(sample)

    dataset = EvaluationDataset(samples=ragas_samples)
    evaluator_llm = get_evaluator_llm()

    metrics = [
        ContextPrecisionWithoutReference(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    print("\n⚙️  Running retrieval evaluation...")
    results = evaluate(dataset=dataset, metrics=metrics)
    return results


def print_retrieval_results(aggregate: Dict, num_samples: int):
    """Print retrieval-specific results to console."""
    print("\n" + "=" * 60)
    print("🔍 RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {num_samples}")
    print("-" * 60)

    for metric, score in aggregate.items():
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        if score >= 0.8:
            status = "✅"
        elif score >= 0.6:
            status = "🟡"
        else:
            status = "🔴"
        print(f"  {status} {metric:<28s} {bar} {score:.4f}")

    print("=" * 60)

    # Diagnostic advice
    print("\n📋 Diagnostic Guide:")
    ctx_prec = aggregate.get("context_precision", 0)
    ctx_rec = aggregate.get("context_recall", 0)

    if ctx_prec < 0.6:
        print("  ⚠ Low Context Precision → Too much irrelevant text retrieved.")
        print("    Try: Increase score threshold (currently 0.2 in query.py)")
        print("    Try: Reduce TOP_K or FINAL_K in config.py")
    if ctx_rec < 0.6:
        print("  ⚠ Low Context Recall → Missing important information.")
        print("    Try: Increase TOP_K to fetch more candidates")
        print("    Try: Reduce CHUNK_SIZE for finer-grained chunks")
    if ctx_prec >= 0.8 and ctx_rec >= 0.8:
        print("  ✅ Retrieval is strong! If answer quality is low,")
        print("     the issue is in the generation (prompt/LLM) side.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval component in isolation")
    parser.add_argument("--testset", type=str, required=True, help="Path to test set JSON file")
    parser.add_argument("--session-id", type=str, required=True, help="Session ID of an ingested document")
    args = parser.parse_args()

    print("=" * 60)
    print("🔍 DevDocs-RAG — Retrieval Component Evaluation")
    print("=" * 60)

    # ── Load test set ─────────────────────────────────────────
    print(f"\n📂 Loading test set: {args.testset}")
    with open(args.testset, "r", encoding="utf-8") as f:
        testset = json.load(f)

    samples = testset["samples"]
    print(f"   Found {len(samples)} test samples")

    # ── Retrieve contexts for each question ───────────────────
    print(f"\n🔄 Running retrieval for each question...")
    eval_samples = []
    for i, sample in enumerate(samples):
        question = sample["user_input"]
        print(f"  [{i+1}/{len(samples)}] {question[:70]}...")

        try:
            contexts = retrieve_contexts(question, args.session_id)
            eval_samples.append({
                "user_input": question,
                "retrieved_contexts": contexts,
                "expected_output": sample.get("expected_output", ""),
            })
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            continue

    if not eval_samples:
        print("\n❌ No samples processed. Check your session ID.")
        return

    print(f"\n✅ Retrieved contexts for {len(eval_samples)}/{len(samples)} samples")

    # ── Evaluate ──────────────────────────────────────────────
    results = evaluate_retrieval_metrics(eval_samples)

    # ── Extract aggregate ─────────────────────────────────────
    aggregate = {}
    # RAGAS EvaluationResult stores scores in a pandas DataFrame
    try:
        # Access the scores dataframe and compute mean for each metric
        if hasattr(results, 'scores'):
            # EvaluationResult has a scores DataFrame
            for col in results.scores.columns:
                aggregate[col] = round(float(results.scores[col].mean()), 4)
        else:
            print(f"⚠ Warning: Could not find scores attribute in results")
    except Exception as e:
        print(f"⚠ Warning: Could not extract scores from results: {e}")

    # ── Print results ─────────────────────────────────────────
    print_retrieval_results(aggregate, len(eval_samples))

    # ── Save results ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "type": "retrieval_only",
            "testset": args.testset,
            "session_id": args.session_id,
            "num_samples": len(eval_samples),
        },
        "aggregate_scores": aggregate,
    }

    json_path = os.path.join(RESULTS_DIR, f"retrieval_eval_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Results saved to: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
