"""
Evaluation harness contract tests.

The actual semantic evaluation is performed by evaluation/run_evaluation.py.
This test validates the dataset and evaluator outputs.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "dataset.json"


def test_evaluation_dataset_is_valid():
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 10

    required = {"id", "category", "question", "expected_behavior"}
    for item in data:
        assert required.issubset(item), f"Invalid dataset item: {item.get('id')}"
        assert item["question"].strip()


def test_evaluation_script_exists():
    assert (ROOT / "evaluation" / "run_evaluation.py").exists()


def test_evaluation_script_help():
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluation" / "run_evaluation.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
