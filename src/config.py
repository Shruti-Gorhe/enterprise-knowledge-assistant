import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-oss:120b-cloud")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge")
TOP_K = int(os.getenv("TOP_K", "4"))
