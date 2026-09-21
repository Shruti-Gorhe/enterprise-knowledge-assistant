import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


BASE_DIR = Path(__file__).resolve().parents[2]


async def search_via_mcp(question: str):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(BASE_DIR),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "search_enterprise_knowledge",
                arguments={"query": question},
            )

            texts = []

            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)

            return "\n\n".join(texts)


def retriever_agent(state):
    question = state["question"]

    print("\n" + "=" * 60)
    print("NODE 1: RETRIEVER AGENT")
    print("=" * 60)
    print(f"Question: {question}")
    print("Calling MCP tool: search_enterprise_knowledge")

    result = asyncio.run(search_via_mcp(question))

    print("MCP retrieval completed.")

    return {
    "retrieved_context": [result],
    "sources": [
        {
            "source": "MCP: search_enterprise_knowledge",
            "page": "N/A"
        }
    ],
    }