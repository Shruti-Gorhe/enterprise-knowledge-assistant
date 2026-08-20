from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(data_dir="data"):
    documents = []
    for pdf_path in Path(data_dir).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number + 1,
                })
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for doc in documents:
        for text in splitter.split_text(doc["text"]):
            chunks.append({
                "text": text,
                "source": doc["source"],
                "page": doc["page"],
            })
    return chunks
