import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(tmp_path: Path, *args: str) -> str:
    env = {**os.environ, "PYTHONPATH": "."}
    result = subprocess.run(
        [sys.executable, "-m", "clawmory_rlm.cli", *args],
        cwd=tmp_path.parent.parent / "clawmory-rlm",
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def test_cli_roundtrip_persists_and_restores_memory(tmp_path: Path):
    repo_root = Path("/Users/sayedraheel/Desktop/Research_and_Books/testrepo/clawmory-rlm")
    history = tmp_path / "history.jsonl"
    structured = tmp_path / "structured.json"

    def run(*args: str) -> str:
        env = {**os.environ, "PYTHONPATH": "."}
        result = subprocess.run(
            [sys.executable, "-m", "clawmory_rlm.cli", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return result.stdout.strip()

    assert "No relevant prior-session memory" in run(
        "--history", str(history),
        "--structured-memory", str(structured),
        "start-session", "Resume", "roundtrip",
    )
    run("--history", str(history), "log", "user", "Final decision: use FastAPI.")
    run("--history", str(history), "log", "assistant", "Next step: connect the runner.")
    summary = run(
        "--history", str(history),
        "--structured-memory", str(structured),
        "end-session", "--label", "roundtrip",
    )
    assert "FastAPI" in summary

    dumped = json.loads(run("--structured-memory", str(structured), "dump-structured"))
    assert dumped["facts"]
    assert dumped["open_loops"]

    resumed = run(
        "--history", str(history),
        "--structured-memory", str(structured),
        "start-session", "Resume", "roundtrip", "again",
    )
    assert "FastAPI" in resumed
    assert "connect the runner" in resumed


def test_cli_namespaces_are_isolated(tmp_path: Path):
    repo_root = Path("/Users/sayedraheel/Desktop/Research_and_Books/testrepo/clawmory-rlm")
    history = tmp_path / "history.jsonl"
    structured = tmp_path / "structured.json"

    def run(*args: str) -> str:
        env = {**os.environ, "PYTHONPATH": "."}
        result = subprocess.run(
            [sys.executable, "-m", "clawmory_rlm.cli", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return result.stdout.strip()

    run("--namespace", "alpha", "--history", str(history), "--structured-memory", str(structured), "start-session", "Alpha")
    run("--history", str(history), "log", "user", "Final decision: use FastAPI for alpha.")
    run("--history", str(history), "log", "assistant", "Next step: alpha task.")
    run("--namespace", "alpha", "--history", str(history), "--structured-memory", str(structured), "end-session", "--label", "alpha")

    run("--namespace", "beta", "--history", str(history), "--structured-memory", str(structured), "start-session", "Beta")
    run("--history", str(history), "log", "user", "Final decision: use Django for beta.")
    run("--history", str(history), "log", "assistant", "Next step: beta task.")
    run("--namespace", "beta", "--history", str(history), "--structured-memory", str(structured), "end-session", "--label", "beta")

    alpha = run("--namespace", "alpha", "--history", str(history), "--structured-memory", str(structured), "start-session", "Resume", "alpha")
    beta = run("--namespace", "beta", "--history", str(history), "--structured-memory", str(structured), "start-session", "Resume", "beta")

    assert "FastAPI for alpha" in alpha
    assert "Django for beta" not in alpha
    assert "Django for beta" in beta
    assert "FastAPI for alpha" not in beta
