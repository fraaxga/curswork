def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    return int(any(chunk_id in relevant_ids for chunk_id in retrieved_ids[:k]))

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    found = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant_ids)
    return found / len(relevant_ids)

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    found = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant_ids)
    return found / k

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_single_query(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> dict:
    return {
        "hit@1": hit_at_k(retrieved_ids, relevant_ids, 1),
        "hit@3": hit_at_k(retrieved_ids, relevant_ids, 3),
        "hit@5": hit_at_k(retrieved_ids, relevant_ids, 5),
        "recall@1": recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall@3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "precision@5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids),
    }

def summarize_retrieval_results(rows: list[dict]) -> dict:
    if not rows:
        return {}
    metric_names = [
        "hit@1",
        "hit@3",
        "hit@5",
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@5",
        "rr",
    ]
    summary = {}
    for metric in metric_names:
        output_name = "mrr" if metric == "rr" else metric
        summary[output_name] = sum(row[metric] for row in rows) / len(rows)
    return summary