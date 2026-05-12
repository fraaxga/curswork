import pandas as pd

def summarize_generation_annotations(annotation_path: str) -> dict:
    df = pd.read_csv(annotation_path)
    required_columns = [
        "correctness",
        "groundedness",
        "citation_accuracy",
        "hallucination",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"missing column in annotation file: {column}")
    summary = {
        "correctness_avg": df["correctness"].mean(),
        "groundedness": df["groundedness"].mean(),
        "citation_accuracy": df["citation_accuracy"].mean(),
        "hallucination_rate": df["hallucination"].mean(),
        "n": len(df),
    }
    return summary