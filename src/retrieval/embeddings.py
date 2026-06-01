from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    def _prepare_documents(self, texts: list[str]) -> list[str]:
        if self._is_e5():
            return [f"passage: {text}" for text in texts]
        return texts

    def _prepare_query(self, query: str) -> str:
        if self._is_e5():
            return f"query: {query}"
        return query

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        prepared_texts = self._prepare_documents(texts)
        embeddings = self.model.encode(
            prepared_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        prepared_query = self._prepare_query(query)
        embedding = self.model.encode(
            [prepared_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.tolist()
    