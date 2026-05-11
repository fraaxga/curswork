import re

def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()

def clean_text_advanced(text: str) -> str:
    text = clean_text_basic(text)
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def clean_pages(pages: list[dict], mode: str = "advanced") -> list[dict]:
    cleaned_pages = []
    for page in pages:
        raw_text = page["text"]
        if mode == "raw":
            cleaned_text = raw_text
        elif mode == "basic":
            cleaned_text = clean_text_basic(raw_text)
        elif mode == "advanced":
            cleaned_text = clean_text_advanced(raw_text)
        else:
            raise ValueError(f"Unknown cleaning mode: {mode}")
        cleaned_pages.append({**page, "text": cleaned_text,})
    return cleaned_pages