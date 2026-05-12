from pathlib import Path
import pandas as pd

from src.config import load_config
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.evaluation.evaluate_retrieval import evaluate_retriever_on_qa

def main():
    config = load_config()
    test_path = config["paths"]["test_set"]
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    top_k = config["retrieval"]["top_k"]
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
    qa_df = pd.read_csv(test_path)
    detailed_df, summary = evaluate_retriever_on_qa(
        retriever=retriever,
        qa_df=qa_df,
        top_k=top_k,
    )
    detailed_path = results_dir / "retrieval_detailed.csv"
    summary_path = results_dir / "retrieval_summary.csv"
    detailed_df.to_csv(detailed_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print("ret sum:")
    print(pd.DataFrame([summary]))
if __name__ == "__main__":
    main()