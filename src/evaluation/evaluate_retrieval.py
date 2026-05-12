import time
import pandas as pd

from src.evaluation.retrieval_metrics import (
    evaluate_single_query,
    summarize_retrieval_results,
)

def evaluate_retriever_on_qa(
    retriever,
    qa_df: pd.DataFrame,
    top_k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for _, row in qa_df.iterrows():
        question = row["question"]
        expected_chunk_id = row["expected_chunk_id"]
        start_time = time.time()
        retrieved = retriever.retrieve(question, top_k=top_k)
        latency = time.time() - start_time
        retrieved_ids = [item["chunk_id"] for item in retrieved]
        metrics = evaluate_single_query(
            retrieved_ids=retrieved_ids,
            relevant_ids={expected_chunk_id},
        )
        rows.append(
            {
                "question_id": row["question_id"],
                "question": question,
                "expected_chunk_id": expected_chunk_id,
                "retrieved_ids": "|".join(retrieved_ids),
                "latency_sec": latency,
                **metrics,
            }
        )
    detailed_df = pd.DataFrame(rows)
    summary = summarize_retrieval_results(rows)
    summary["avg_latency_sec"] = detailed_df["latency_sec"].mean()
    return detailed_df, summary