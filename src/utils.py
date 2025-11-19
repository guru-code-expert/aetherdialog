import re

def clean_text(text: str) -> str:
    """Remove non-alphabetic characters and lowercase the text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()