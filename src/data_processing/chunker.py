import re

def extract_section_id(text: str) -> str | None:
    match = re.search(r"(^|\n)(\d+(\.\d+)+)", text)
    if match:
        return match.group(2)
    return None

def fixed_size_chunks(
    pages: list[dict],
    document_id: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    global_index = 0
    for page_data in pages:
        page = page_data["page"]
        text = page_data["text"]
        start = 0
        local_index = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{document_id}_p{page}_c{local_index:03d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "page": page,
                        "text": chunk_text,
                        "chunk_index": global_index,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunk_method": "fixed",
                        "source": source,
                        "section_id": extract_section_id(chunk_text),
                        "section_title": None,
                    }
                )
                global_index += 1
                local_index += 1
            start += chunk_size - chunk_overlap
    return chunks

def paragraph_chunks(
    pages: list[dict],
    document_id: str,
    source: str,
    max_chars: int = 1200,
) -> list[dict]:
    chunks = []
    global_index = 0
    for page_data in pages:
        page = page_data["page"]
        paragraphs = [
            paragraph.strip()
            for paragraph in page_data["text"].split("\n")
            if paragraph.strip()
        ]
        buffer = ""
        local_index = 0
        for paragraph in paragraphs:
            candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunk_id = f"{document_id}_p{page}_c{local_index:03d}"
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "page": page,
                            "text": buffer,
                            "chunk_index": global_index,
                            "chunk_size": max_chars,
                            "chunk_overlap": 0,
                            "chunk_method": "paragraph",
                            "source": source,
                            "section_id": extract_section_id(buffer),
                            "section_title": None,
                        }
                    )
                    global_index += 1
                    local_index += 1
                buffer = paragraph
        if buffer:
            chunk_id = f"{document_id}_p{page}_c{local_index:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page": page,
                    "text": buffer,
                    "chunk_index": global_index,
                    "chunk_size": max_chars,
                    "chunk_overlap": 0,
                    "chunk_method": "paragraph",
                    "source": source,
                    "section_id": extract_section_id(buffer),
                    "section_title": None,
                }
            )
            global_index += 1
    return chunks

def recursive_chunks(pages: list[dict], document_id: str, source: str, chunk_size: int = 800, chunk_overlap: int = 100,) -> list[dict]:
    intermediate_pages = []

    for page in pages:
        text = page["text"]
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        rebuilt = []
        buffer = ""
        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if buffer:
                    rebuilt.append(buffer)
                    buffer = ""
                step = chunk_size - chunk_overlap
                for start in range(0, len(paragraph), step):
                    rebuilt.append(paragraph[start:start + chunk_size])
            else:
                candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
                if len(candidate) <= chunk_size:
                    buffer = candidate
                else:
                    rebuilt.append(buffer)
                    buffer = paragraph
        if buffer:
            rebuilt.append(buffer)
        intermediate_pages.append(
            {
                "page": page["page"],
                "text": "\n\n".join(rebuilt),
            }
        )
    return paragraph_chunks(
        pages=intermediate_pages,
        document_id=document_id,
        source=source,
        max_chars=chunk_size,
    )

def build_chunks(
    pages: list[dict],
    document_id: str,
    source: str,
    method: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    if method == "fixed":
        return fixed_size_chunks(
            pages=pages,
            document_id=document_id,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if method == "paragraph":
        return paragraph_chunks(
            pages=pages,
            document_id=document_id,
            source=source,
            max_chars=chunk_size,
        )
    if method == "recursive":
        return recursive_chunks(
            pages=pages,
            document_id=document_id,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    raise ValueError(f"Unknown chunking method: {method}")