"""
Agent/graph contract tests.

These tests verify the documented agent modules and LangGraph structure without
calling a paid API or requiring a live LLM.
"""
from pathlib import Path
import importlib
import re

ROOT = Path(__file__).resolve().parents[1]


def test_agent_modules_exist():
    expected = [
        "src/agents/retriever_agent.py",
        "src/agents/response_agent.py",
        "src/agents/evaluator_agent.py",
        "src/graph.py",
        "src/state.py",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"Missing agent/graph files: {missing}"


def test_agent_modules_import():
    expected = {
        "src/agents/retriever_agent.py": "retriever_agent",
        "src/agents/response_agent.py": "response_agent",
        "src/agents/evaluator_agent.py": "evaluator_agent",
    }

    failures = []

    for file_path, function_name in expected.items():
        path = ROOT / file_path

        if not path.exists():
            failures.append(f"{file_path}: file does not exist")
            continue

        text = path.read_text(encoding="utf-8")

        if f"def {function_name}" not in text:
            failures.append(
                f"{file_path}: expected function "
                f"{function_name}() not found"
            )

    assert not failures, "Agent contract failures:\n" + "\n".join(failures)


def test_graph_mentions_expected_nodes():
    graph_file = ROOT / "src" / "graph.py"
    text = graph_file.read_text(encoding="utf-8").lower()

    required_concepts = ["retriev", "response", "evaluat"]
    missing = [x for x in required_concepts if x not in text]
    assert not missing, f"LangGraph does not visibly contain concepts: {missing}"


def test_documented_pipeline_order_is_present():
    text = (ROOT / "src" / "graph.py").read_text(encoding="utf-8").lower()
    positions = []
    for term in ("retriev", "response", "evaluat"):
        pos = text.find(term)
        assert pos >= 0, f"Could not find {term} node reference"
        positions.append(pos)
    assert positions == sorted(positions), (
        "Expected graph.py to document/construct the order "
        "Retriever -> Response -> Evaluator. If your implementation "
        "uses conditional edges, review this contract test."
    )
