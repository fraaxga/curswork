from pathlib import Path
import csv
import json

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import RAGGenerator
from src.evaluation.evaluate_generation import build_generation_annotation_template


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    load_dotenv()

    config = load_config()

    test_path = config["paths"]["test_set"]
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    top_k = config["retrieval"]["top_k"]

    qa_df = pd.read_csv(test_path)

    embedding_model = EmbeddingModel(config["embedding"]["model_name"])

    vector_store = ChromaVectorStore(
        collection_name=config["vector_store"]["collection_name"],
        persist_directory=config["vector_store"]["persist_directory"],
        reset_collection=False,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=top_k,
    )

    generator = RAGGenerator(
        model_name=config["generation"]["model_name"],
        temperature=config["generation"]["temperature"],
        prompt_type="strict",
    )

    generated_answers = []

    for _, row in qa_df.iterrows():
        question_id = row["question_id"]
        question = row["question"]

        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [item["chunk_id"] for item in retrieved]

        answer = generator.generate(
            question=question,
            retrieved_chunks=retrieved,
        )

        generated_answers.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_answer": row.get("expected_answer", ""),
                "expected_chunk_id": row.get("expected_chunk_id", ""),
                "generated_answer": answer,
                "retrieved_ids": "|".join(retrieved_ids),
            }
        )

    generated_df = pd.DataFrame(generated_answers)

    generated_csv_path = results_dir / "generation_raw_outputs.csv"
    generated_jsonl_path = results_dir / "generation_raw_outputs.jsonl"

    generated_df.to_csv(
        generated_csv_path,
        index=False,
        quoting=csv.QUOTE_ALL,
    )

    save_jsonl(generated_answers, generated_jsonl_path)

    annotation_df = build_generation_annotation_template(
        qa_df=qa_df,
        generated_answers=generated_answers,
    )

    annotation_path = results_dir / "generation_annotation_template.csv"

    annotation_df.to_csv(
        annotation_path,
        index=False,
        quoting=csv.QUOTE_ALL,
    )


if __name__ == "__main__":
    main()