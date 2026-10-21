import re


def clean_transcript(text: str) -> str:
    """Standardizes transcripts: strips brackets, punctuations, and normalizes spacing."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    text = text.upper()
    text = re.sub(r"\s+", " ", text).strip()
    return text