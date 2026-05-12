from openai import OpenAI
from src.generation.prompt import STRICT_RAG_PROMPT, BASIC_RAG_PROMPT

class RAGGenerator:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_type: str = "strict",
    ):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_type = prompt_type
    @staticmethod
    def build_context(retrieved_chunks: list[dict]) -> str:
        parts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})
            page = metadata.get("page", "")
            section_id = metadata.get("section_id", "")
            source_line = f"Источник {index}: страница {page}"
            if section_id:
                source_line += f", пункт {section_id}"
            parts.append(
                f"{source_line}\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(parts)
    def build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        context = self.build_context(retrieved_chunks)
        if self.prompt_type == "strict":
            template = STRICT_RAG_PROMPT
        elif self.prompt_type == "basic":
            template = BASIC_RAG_PROMPT
        return template.format(
            context=context,
            question=question,
        )
    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:
        prompt = self.build_prompt(question, retrieved_chunks)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.choices[0].message.content