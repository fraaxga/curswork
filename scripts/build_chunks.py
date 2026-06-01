from src.config import load_config
from src.data_processing.pdf_loader import load_pdf_by_pages
from src.data_processing.cleaner import clean_pages
from src.data_processing.chunker import build_chunks
from src.data_processing.build_dataset import (
    save_jsonl,
    add_document_metadata,
)


def main():
    config = load_config()
    pdf_path = config["paths"]["raw_pdf"]
    documents_path = config["paths"]["documents"]
    chunks_path = config["paths"]["chunks"]
    document_id = config["document"]["document_id"]
    source = config["document"]["source"]
    version = config["document"]["version"]
    effective_from = config["document"]["effective_from"]
    preprocessing_mode = config["preprocessing"]["mode"]
    chunk_method = config["chunking"]["method"]
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]
    pages = load_pdf_by_pages(pdf_path)
    cleaned_pages = clean_pages(pages, mode=preprocessing_mode)
    documents = add_document_metadata(
        pages=cleaned_pages,
        document_id=document_id,
        source=source,
        version=version,
        effective_from=effective_from,
    )
    save_jsonl(documents, documents_path)
    chunks = build_chunks(
        pages=cleaned_pages,
        document_id=document_id,
        source=source,
        method=chunk_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"saving {len(chunks)} chunks")
    save_jsonl(chunks, chunks_path)
if __name__ == "__main__":
    main()