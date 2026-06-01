from pathlib import Path
import fitz


def load_pdf_by_pages(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)
    pages = []
    for page_index, page in enumerate(document, start=1):
        text = page.get_text("text")
        pages.append(
            {
                "page": page_index,
                "text": text,
            }
        )
    document.close()
    return pages