from pathlib import Path

from mcp.server import MCPServer

from src.rag.retriever import retrieve_documents


mcp = MCPServer("enterprise-knowledge-assistant")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@mcp.tool()
def search_enterprise_knowledge(query: str) -> str:
    result = retrieve_documents(query)

    documents = result[0] if isinstance(result, tuple) else result

    output = []

    for document in documents:
        if hasattr(document, "page_content"):
            content = document.page_content
            metadata = document.metadata or {}

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

        else:
            content = str(document)
            source = "Retrieved Knowledge"
            page = "Unknown"

        output.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {content}"
        )

    if not output:
        return "No relevant documents found."

    return "\n\n---\n\n".join(output)


@mcp.tool()
def get_document_sources() -> str:
    if not DATA_DIR.exists():
        return f"Data directory not found: {DATA_DIR}"

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        return "No PDF documents found."

    return "\n".join(
        f"- {pdf.name}"
        for pdf in pdf_files
    )


if __name__ == "__main__":
    mcp.run()