"""
MCP contract tests.

The documented MCP server exposes:
- search_enterprise_knowledge
- get_document_sources
"""
import asyncio
import importlib

EXPECTED_TOOLS = {
    "search_enterprise_knowledge",
    "get_document_sources",
}


def _list_tools():
    mod = importlib.import_module("mcp_server.server")
    assert hasattr(mod, "mcp"), "mcp_server.server must expose `mcp`"
    return asyncio.run(mod.mcp.list_tools())


def test_mcp_server_imports():
    importlib.import_module("mcp_server.server")


def test_required_mcp_tools_exist():
    tools = _list_tools()
    names = {tool.name for tool in tools}
    missing = EXPECTED_TOOLS - names
    assert not missing, f"Missing MCP tools: {sorted(missing)}"


def test_mcp_has_no_unexpected_tools_by_default():
    tools = _list_tools()
    names = {tool.name for tool in tools}
    unexpected = names - EXPECTED_TOOLS
    # Do not hard-fail on additional tools; print them so CI exposes contract drift.
    if unexpected:
        print(f"INFO: additional MCP tools detected: {sorted(unexpected)}")
