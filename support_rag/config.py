"""Central configuration for the customer-support RAG app.

All tunables (models, paths, MLflow target, retrieval defaults, prompts) live here so
the rest of the package contains logic, not magic constants.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # make OPENAI_API_KEY (etc.) available package-wide

# --- MLflow ---
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "customer-support-rag"

# --- Models ---
DENSE_MODEL = "text-embedding-3-small"  # OpenAI dense embeddings
CHAT_MODEL = "gpt-4o-mini"              # generator
JUDGE_MODEL = "openai:/gpt-4o-mini"     # MLflow LLM-judge scorers

# --- Knowledge base (hand-authored Markdown corpus) ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSIST_DIR = DATA_DIR / "chroma"
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
DENSE_COLLECTION = "nimbus_support"
SPARSE_MODELS = {
    "bm25": "Qdrant/bm25",
}

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval defaults ---
DEFAULT_MODE = "hybrid"     # dense | sparse | hybrid
DEFAULT_SPARSE = "bm25"     # the only sparse model
DEFAULT_K = 4
CANDIDATE_LIMIT = 50        # candidates pulled per retriever before fusion / top-k
HYBRID_WEIGHTS = (0.7, 0.3)  # (dense, sparse) for hybrid RRF
RRF_K = 60                  # reciprocal-rank-fusion constant

# --- Prompts ---
SYSTEM_PROMPT = (
    "You are a helpful, polite customer-support assistant. "
    "Answer the customer's question using ONLY the support knowledge in the context. "
    "Give clear, step-by-step guidance when relevant. If the context does not contain "
    "the answer, say you don't have that information and suggest contacting a human "
    "support agent. Never invent policies, prices, fees, or steps."
)

SUPPORT_GUIDELINE = (
    "The response must be polite and professional, must not invent policies, prices, "
    "fees, or steps that are not supported by the provided knowledge, and should "
    "suggest contacting a human support agent when it cannot answer."
)
