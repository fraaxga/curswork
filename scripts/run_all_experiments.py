from pathlib import Path
import shutil

import pandas as pd

from src.config import load_config
from src.data_processing.pdf_loader import load_pdf_by_pages
from src.data_processing.cleaner import clean_pages
from src.data_processing.chunker import build_chunks
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.bm_25_ret import BM25Retriever
from src.evaluation.evaluate_retrieval import evaluate_retriever_on_qa

def evaluate_config(
    experiment_name: str,
    pages: list[dict],
    qa_df: pd.DataFrame,
    document_id: str,
    source: str,
    preprocessing_mode: str,
    chunk_method: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    top_k: int,
    use_reranker: bool = False,
    reranker_model_name: str | None = None,
    initial_top_k: int = 10,
    use_bm25: bool = False,
) -> dict:
    cleaned_pages = clean_pages(pages, mode=preprocessing_mode)
    chunks = build_chunks(
        pages=cleaned_pages,
        document_id=document_id,
        source=source,
        method=chunk_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if use_bm25:
        retriever = BM25Retriever(chunks)
        detailed_df, summary = evaluate_retriever_on_qa(
            retriever=retriever,
            qa_df=qa_df,
            top_k=top_k,
        )
    else:
        persist_dir = f".chroma_experiments/{experiment_name}"
        if Path(persist_dir).exists():
            shutil.rmtree(persist_dir)
        embedding_model = EmbeddingModel(embedding_model_name)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.encode_texts(texts)
        vector_store = ChromaVectorStore(
            collection_name="experiment_collection",
            persist_directory=persist_dir,
            reset_collection=True,
        )
        vector_store.add_chunks(chunks, embeddings)
        base_retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=initial_top_k if use_reranker else top_k,
        )
        if use_reranker:
            reranker = CrossEncoderReranker(reranker_model_name)
            class RerankedRetriever:
                def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
                    initial = base_retriever.retrieve(question, top_k=initial_top_k)
                    return reranker.rerank(question, initial, top_k=top_k)

            retriever = RerankedRetriever()
        else:
            retriever = base_retriever
        detailed_df, summary = evaluate_retriever_on_qa(
            retriever=retriever,
            qa_df=qa_df,
            top_k=top_k,
        )
    summary.update(
        {
            "experiment_name": experiment_name,
            "preprocessing": preprocessing_mode,
            "chunk_method": chunk_method,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": "BM25" if use_bm25 else embedding_model_name,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "n_chunks": len(chunks),
        }
    )
    return summary

def main():
    config = load_config()
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pages = load_pdf_by_pages(config["paths"]["raw_pdf"])
    qa_df = pd.read_csv(config["paths"]["test_set"])
    document_id = config["document"]["document_id"]
    source = config["document"]["source"]
    top_k = 5
    all_results = []
    preprocessing_configs = ["raw", "basic", "advanced"]
    for mode in preprocessing_configs:
        result = evaluate_config(
            experiment_name=f"preprocessing_{mode}",
            pages=pages,
            qa_df=qa_df,
            document_id=document_id,
            source=source,
            preprocessing_mode=mode,
            chunk_method="paragraph",
            chunk_size=1000,
            chunk_overlap=0,
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            top_k=top_k,
        )
        all_results.append(result)
    chunking_configs = [
        ("fixed", 500, 50),
        ("fixed", 800, 100),
        ("fixed", 1200, 150),
        ("paragraph", 1000, 0),
        ("recursive", 800, 100),
    ]
    for method, size, overlap in chunking_configs:
        result = evaluate_config(
            experiment_name=f"chunking_{method}_{size}_{overlap}",
            pages=pages,
            qa_df=qa_df,
            document_id=document_id,
            source=source,
            preprocessing_mode="basic",
            chunk_method=method,
            chunk_size=size,
            chunk_overlap=overlap,
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            top_k=top_k,
        )
        all_results.append(result)
    embedding_models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
    ]
    for model_name in embedding_models:
        result = evaluate_config(
            experiment_name=f"embedding_{model_name.split('/')[-1]}",
            pages=pages,
            qa_df=qa_df,
            document_id=document_id,
            source=source,
            preprocessing_mode="basic",
            chunk_method="paragraph",
            chunk_size=1000,
            chunk_overlap=0,
            embedding_model_name=model_name,
            top_k=top_k,
        )
        all_results.append(result)
    bm25_result = evaluate_config(
        experiment_name="bm25_baseline",
        pages=pages,
        qa_df=qa_df,
        document_id=document_id,
        source=source,
        preprocessing_mode="basic",
        chunk_method="paragraph",
        chunk_size=1000,
        chunk_overlap=0,
        embedding_model_name="BM25",
        top_k=top_k,
        use_bm25=True,
    )
    all_results.append(bm25_result)
    reranker_result = evaluate_config(
        experiment_name="reranker_cross_encoder",
        pages=pages,
        qa_df=qa_df,
        document_id=document_id,
        source=source,
        preprocessing_mode="basic",
        chunk_method="paragraph",
        chunk_size=1000,
        chunk_overlap=0,
        embedding_model_name="intfloat/multilingual-e5-small",
        top_k=top_k,
        use_reranker=True,
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_top_k=10,
    )
    all_results.append(reranker_result)
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "all_retrieval_experiments.csv"
    results_df.to_csv(output_path, index=False)
    print(results_df)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()