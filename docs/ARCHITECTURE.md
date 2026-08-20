# Architecture

User -> LangGraph
- Retriever Agent -> ChromaDB + HuggingFace embeddings
- Response Agent -> Ollama / Qwen3
- Evaluator Agent -> RAGAS + Ollama + HuggingFace embeddings

Streamlit -> LangGraph
MCP -> local read-only knowledge tools

All inference is local.
