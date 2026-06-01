import re
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[а-яёa-z0-9]+", text)

class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(question)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]
        results = []
        for index in ranked_indices:
            chunk = self.chunks[index]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": {
                        "document_id": chunk.get("document_id", ""),
                        "page": chunk.get("page", ""),
                        "source": chunk.get("source", ""),
                        "section_id": chunk.get("section_id") or "",
                        "section_title": chunk.get("section_title") or "",
                    },
                    "bm25_score": float(scores[index]),
                }
            )
        return results