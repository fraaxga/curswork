from src.config import load_config
from src.data_processing.build_dataset import load_jsonl
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore

def main():
    config = load_config()
    chunks_path = config["paths"]["chunks"]
    embedding_model_name = config["embedding"]["model_name"]
    collection_name = config["vector_store"]["collection_name"]
    persist_directory = config["vector_store"]["persist_directory"]
    chunks = load_jsonl(chunks_path)
    embedding_model = EmbeddingModel(embedding_model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode_texts(texts)
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        reset_collection=True,
    )
    vector_store.add_chunks(chunks, embeddings)
    print(f"chroma collection size: {vector_store.count()}")
if __name__ == "__main__":
    main()