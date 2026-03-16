import json
from pathlib import Path

from clawmory_rlm import AgentSession


FIXTURES_PATH = Path(__file__).parent / "fixtures" / "benchmark_transcripts.json"


def _load_cases():
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))["cases"]


def _materialize_case(tmp_path: Path, case: dict) -> str:
    history = tmp_path / f"{case['name']}_history.jsonl"
    structured = tmp_path / f"{case['name']}_structured.json"

    for session_data in case["sessions"]:
        with AgentSession(
            goal=session_data["label"],
            history_path=history,
            structured_memory_path=structured,
            session_label=session_data["label"],
            namespace=case["namespace"],
        ) as session:
            for role, content in session_data["turns"]:
                session.log_turn(role, content)

    with AgentSession(
        goal=case["goal"],
        history_path=history,
        structured_memory_path=structured,
        session_label="resume",
        namespace=case["namespace"],
    ) as resumed:
        return resumed.state.system_prompt


def test_memory_benchmark_cases(tmp_path: Path):
    failures = []
    scores = []

    for case in _load_cases():
        prompt = _materialize_case(tmp_path, case)

        hit_count = sum(1 for text in case["expected_substrings"] if text in prompt)
        miss_count = sum(1 for text in case["forbidden_substrings"] if text in prompt)
        score = hit_count - miss_count
        scores.append((case["name"], score, len(case["expected_substrings"])))

        missing = [text for text in case["expected_substrings"] if text not in prompt]
        forbidden = [text for text in case["forbidden_substrings"] if text in prompt]
        if missing or forbidden:
            failures.append(
                {
                    "name": case["name"],
                    "missing": missing,
                    "forbidden": forbidden,
                    "prompt": prompt,
                }
            )

    assert not failures, f"Benchmark failures: {failures}"
    assert all(score >= expected for _, score, expected in scores)
