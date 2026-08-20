from src.rag.retriever import retrieve_documents

def retriever_agent(state):
    question = state["question"]
    print("\n" + "=" * 60)
    print("NODE 1: RETRIEVER AGENT")
    print("=" * 60)
    print(f"Question: {question}")
    contexts, sources = retrieve_documents(question)
    print(f"Retrieved {len(contexts)} documents.")
    return {"retrieved_context": contexts, "sources": sources}
