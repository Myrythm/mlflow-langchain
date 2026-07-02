"""Demo: ask the default (hybrid) customer-support agent a few questions.

    uv run python main.py

Each answer is traced into the `customer-support-rag` experiment at
http://127.0.0.1:5000.
"""

import sys

from support_rag.config import EXAMPLE_QUESTIONS
from support_rag.generation import get_default_answer
from support_rag.tracing import setup_tracing


def main() -> None:
    # KB answers contain Unicode (e.g. "Settings → Integrations"); make stdout
    # UTF-8 so printing them doesn't crash on a Windows cp1252 console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_tracing()
    answer = get_default_answer()
    for question in EXAMPLE_QUESTIONS:
        print(f"\nQ: {question}")
        print(f"A: {answer(question)}")


if __name__ == "__main__":
    main()
