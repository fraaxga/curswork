from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
    def rerank(
        self,
        question: str,
        retrieved_chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not retrieved_chunks:
            return []
        pairs = [(question, chunk["text"]) for chunk in retrieved_chunks]
        scores = self.model.predict(pairs)
        reranked = []
        for chunk, score in zip(retrieved_chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["reranker_score"] = float(score)
            reranked.append(chunk_copy)
        reranked = sorted(
            reranked,
            key=lambda item: item["reranker_score"],
            reverse=True,
        )
        return reranked[:top_k]