from langchain_ollama import ChatOllama
from src.config import OLLAMA_BASE_URL, LLM_MODEL

def response_agent(state):
    print("\n" + "=" * 60)
    print("NODE 2: RESPONSE AGENT")
    print("=" * 60)

    context_text = "\n\n---\n\n".join(state.get("retrieved_context", []))
    prompt = f"""You are an enterprise knowledge assistant.

Answer the user's question using ONLY the supplied context.
If the context does not contain the answer, say the information is not available.

Question:
{state["question"]}

Context:
{context_text}
"""

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    answer = llm.invoke(prompt).content
    print("Generated answer:")
    print(answer)
    return {"answer": answer}
