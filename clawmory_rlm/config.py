import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "history.jsonl"
DEFAULT_STRUCTURED_MEMORY_PATH = PROJECT_ROOT / "structured_memory.json"
DEFAULT_MODEL = os.getenv("CLAWMORY_MODEL", "gpt-oss:20b")
DEFAULT_RLM_TIMEOUT_S = int(os.getenv("CLAWMORY_RLM_TIMEOUT_S", "20"))


def resolve_history_path(path: str | os.PathLike | None = None) -> Path:
    if path is None:
        return DEFAULT_HISTORY_PATH
    return Path(path).expanduser().resolve()
