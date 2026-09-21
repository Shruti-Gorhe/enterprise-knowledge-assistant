from src.rag.loader import load_pdfs, chunk_documents
from src.rag.vectorstore import create_vectorstore

def main():
    print("=" * 60)
    print("ENTERPRISE KNOWLEDGE ASSISTANT - INGESTION")
    print("=" * 60)
    docs = load_pdfs("data")
    print(f"Loaded {len(docs)} pages.")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    create_vectorstore(chunks)
    print("Vector database created successfully.")

if __name__ == "__main__":
    main()
