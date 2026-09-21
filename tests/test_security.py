"""
Security regression tests for an enterprise RAG/agentic system.

These are repository-level guardrail checks. They do not claim that the LLM is
secure merely because these tests pass.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

INJECTION_PATTERNS = [
    r"ignore\s+all\s+previous\s+instructions",
    r"reveal\s+the\s+system\s+prompt",
    r"send\s+.*confidential",
    r"disregard\s+.*instructions",
]


def test_no_hardcoded_common_secret_patterns():
    candidates = []
    for folder in ("src", "mcp_server", "scripts"):
        path = ROOT / folder
        if not path.exists():
            continue
        candidates.extend(p for p in path.rglob("*.py") if ".venv" not in str(p))

    suspicious = []
    secret_regexes = [
        re.compile(r"(?i)api[_-]?key\s*=\s*['\"][^'\"]{12,}['\"]"),
        re.compile(r"(?i)secret\s*=\s*['\"][^'\"]{12,}['\"]"),
        re.compile(r"(?i)password\s*=\s*['\"][^'\"]+['\"]"),
    ]
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for rx in secret_regexes:
            if rx.search(text):
                suspicious.append(str(path.relative_to(ROOT)))
                break

    assert not suspicious, f"Possible hardcoded secret/config value in: {suspicious}"


def test_injection_fixture_is_treated_as_data():
    fixture = ROOT / "evaluation" / "dataset.json"
    assert fixture.exists()
    text = fixture.read_text(encoding="utf-8").lower()

    assert any(re.search(pattern, text) for pattern in INJECTION_PATTERNS), (
        "Security evaluation dataset should include at least one prompt-injection case."
    )


def test_env_file_is_not_tracked_by_this_test_layout():
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return
    text = gitignore.read_text(encoding="utf-8")
    assert ".env" in text, ".gitignore should exclude .env"
