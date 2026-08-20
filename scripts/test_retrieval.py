from src.rag.retriever import retrieve_documents

def main():
    question = input("Enter a retrieval question: ")
    contexts, sources = retrieve_documents(question)
    print(f"\nRetrieved {len(contexts)} documents.")
    for i, (context, source) in enumerate(zip(contexts, sources), 1):
        print("=" * 60)
        print(f"Document {i}")
        print(f"Source: {source['source']}")
        print(f"Page: {source['page']}")
        print(context[:1000])

if __name__ == "__main__":
    main()
