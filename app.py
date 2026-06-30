"""Launch the Gradio customer-support chat UI.

    uv run python app.py

Opens at http://127.0.0.1:7860 (the MLflow tracking server should also be running).
"""

from support_rag.ui import build_demo

if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860)
