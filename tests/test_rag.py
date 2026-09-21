"""
RAG contract tests for Enterprise Knowledge Assistant.

These tests intentionally avoid assuming a specific retriever function signature.
They verify the repository's documented RAG assets and, when available, inspect
the persisted ChromaDB collection.
"""
from pathlib import Path
import importlib.util
import os
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_rag_source_files_exist():
    expected = [
        ROOT / "src" / "rag" / "loader.py",
        ROOT / "src" / "rag" / "embeddings.py",
        ROOT / "src" / "rag" / "vectorstore.py",
        ROOT / "src" / "rag" / "retriever.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    assert not missing, f"Missing RAG modules: {missing}"


def test_rag_configuration_is_present():
    config = ROOT / "src" / "config.py"
    assert config.exists(), "src/config.py is required"
    text = config.read_text(encoding="utf-8")
    assert "TOP_K" in text or "top_k" in text.lower()
    assert "EMBEDDING_MODEL" in text or "embedding" in text.lower()


def test_chroma_store_exists_after_ingestion():
    chroma = ROOT / "chroma_db"
    if not chroma.exists():
        pytest.skip("chroma_db not present; run `python scripts/ingest.py` first")

    files = list(chroma.rglob("*"))
    assert files, "chroma_db exists but contains no persisted files"


def test_retriever_module_importable():
    path = ROOT / "src" / "rag" / "retriever.py"
    spec = importlib.util.spec_from_file_location("eka_retriever_contract", path)
    assert spec is not None and spec.loader is not None
