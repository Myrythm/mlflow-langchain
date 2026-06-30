"""Demo: ask the default (hybrid) customer-support agent a few questions.

    uv run python main.py

Each answer is traced into the `customer-support-rag` experiment at
http://127.0.0.1:5000.
"""

from support_rag.generation import get_default_answer
from support_rag.tracing import setup_tracing

# In-domain questions for the Nimbus support KB.
SAMPLE_QUESTIONS = [
    "How much does the Pro plan cost per member?",
    "Does Nimbus support single sign-on?",
    "How do I connect Slack to Nimbus?",
]


def main() -> None:
    setup_tracing()
    answer = get_default_answer()
    for question in SAMPLE_QUESTIONS:
        print(f"\nQ: {question}")
        print(f"A: {answer(question)}")


if __name__ == "__main__":
    main()
