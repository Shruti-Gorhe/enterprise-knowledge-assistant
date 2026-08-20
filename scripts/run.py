from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph

def main():
    print("=" * 60)
    print("ENTERPRISE KNOWLEDGE ASSISTANT")
    print("=" * 60)

    question = input("\nAsk your question: ").strip()
    if not question:
        return

    result = build_graph().invoke({"question": question})

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"\nQUESTION:\n{question}")
    print(f"\nANSWER:\n{result.get('answer', '')}")

    print("\nSOURCES:")
    for s in result.get("sources", []):
        print(f"- {s.get('source')} (page {s.get('page')})")

    print("\nRAGAS SCORES:")
    print(f"Faithfulness: {result.get('faithfulness', 0):.4f}")
    print(f"Answer Relevancy: {result.get('answer_relevancy', 0):.4f}")
    print(f"Interpretation: {result.get('evaluation', {}).get('interpretation', 'N/A')}")

if __name__ == "__main__":
    main()
