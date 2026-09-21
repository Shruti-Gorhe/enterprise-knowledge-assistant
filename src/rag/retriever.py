from src.config import TOP_K
from src.rag.vectorstore import get_vectorstore

def retrieve_documents(question, k=TOP_K):
    results = get_vectorstore().similarity_search_with_score(question, k=k)
    contexts, sources = [], []
    for doc, score in results:
        contexts.append(doc.page_content)
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "score": float(score),
        })
    return contexts, sources
