import pandas as pd

def build_generation_annotation_template(
    qa_df: pd.DataFrame,
    generated_answers: list[dict],
) -> pd.DataFrame:
    answer_by_id = {
        item["question_id"]: item
        for item in generated_answers
    }
    rows = []
    for _, row in qa_df.iterrows():
        question_id = row["question_id"]
        generated = answer_by_id.get(question_id, {})
        rows.append(
            {
                "question_id": question_id,
                "question": row["question"],
                "expected_answer": row.get("expected_answer", ""),
                "expected_chunk_id": row.get("expected_chunk_id", ""),
                "generated_answer": generated.get("generated_answer", ""),
                "retrieved_ids": generated.get("retrieved_ids", ""),
                "correctness": "",
                "groundedness": "",
                "citation_accuracy": "",
                "hallucination": "",
                "comment": "",
            }
        )
    return pd.DataFrame(rows)