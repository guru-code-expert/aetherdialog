import os
import yaml
from typing import List, Tuple
from pathlib import Path

from .utils import clean_text


def load_conversation_pairs(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load all conversation YAML files from data_dir and extract Q/A pairs.

    Handles both simple [question, answer] and multi-response conversations.
    """
    questions = []
    answers = []

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    for file_path in data_path.glob("*.yml"):
        with open(file_path, "rb") as f:
            docs = yaml.safe_load(f)

        conversations = docs.get("conversations", [])
        for conv in conversations:
            if len(conv) >= 2:
                questions.append(clean_text(str(conv[0])))
                # Join possible multiple replies into one answer
                reply = " ".join(str(r) for r in conv[1:])
                answers.append(clean_text(reply))

    return questions, answers