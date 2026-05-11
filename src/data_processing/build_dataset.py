from pathlib import Path
import json

def save_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_jsonl(input_path: str | Path) -> list[dict]:
    input_path = Path(input_path)
    records = []
    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records

def add_document_metadata(pages: list[dict], document_id: str, source: str, version: str, effective_from: str,) -> list[dict]:
    documents = []
    for page in pages:
        documents.append(
            {
                "document_id": document_id,
                "page": page["page"],
                "text": page["text"],
                "source": source,
                "version": version,
                "effective_from": effective_from,
            }
        )
    return documents