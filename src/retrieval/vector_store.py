import chromadb

class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = ".chroma",
        reset_collection: bool = False,
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        if reset_collection:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadatas.append(
                {
                    "document_id": chunk.get("document_id", ""),
                    "page": int(chunk.get("page", 0)),
                    "source": chunk.get("source", ""),
                    "section_id": chunk.get("section_id") or "",
                    "section_title": chunk.get("section_title") or "",
                    "chunk_method": chunk.get("chunk_method") or "",
                    "chunk_size": int(chunk.get("chunk_size") or 0),
                    "chunk_overlap": int(chunk.get("chunk_overlap") or 0),
                }
            )
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for index in range(len(result["ids"][0])):
            retrieved.append(
                {
                    "chunk_id": result["ids"][0][index],
                    "text": result["documents"][0][index],
                    "metadata": result["metadatas"][0][index],
                    "distance": result["distances"][0][index],
                }
            )
        return retrieved
    def count(self) -> int:
        return self.collection.count()
    