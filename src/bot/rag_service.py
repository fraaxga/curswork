from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import RAGGenerator

class RAGService:
    def __init__(self) -> None:
        load_dotenv()
        self.config = load_config()
        self.embedding_model = EmbeddingModel(
            self.config["embedding"]["model_name"]
        )
        self.vector_store = ChromaVectorStore(
            collection_name=self.config["vector_store"]["collection_name"],
            persist_directory=self.config["vector_store"]["persist_directory"],
            reset_collection=False,
        )
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=self.config["retrieval"]["top_k"],
        )
        self.generator = RAGGenerator(
            model_name=self.config["generation"]["model_name"],
            temperature=self.config["generation"]["temperature"],
            prompt_type="strict",
        )
    def answer_question(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Задайте интересующий Вас вопрос по документу ПОПАТКУС."
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=self.config["retrieval"]["top_k"],
        )
        if not retrieved_chunks:
            return (
                "Мне не удалось найти информацию в документе по вашему запросу, "
                "посетите страницу с контактами службы поддержки: "
                "https://www.hse.ru/contacts.html"
            )
        return self.generator.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )