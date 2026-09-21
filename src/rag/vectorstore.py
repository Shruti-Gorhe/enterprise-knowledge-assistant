from langchain_chroma import Chroma
from src.config import VECTOR_DB_PATH, COLLECTION_NAME
from src.rag.embeddings import get_embeddings

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=VECTOR_DB_PATH,
    )

def create_vectorstore(chunks):
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    return Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
    )
