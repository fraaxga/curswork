from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore

class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        query_embedding = self.embedding_model.encode_query(question)
        return self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
        )