"""Load parametrized test cases from JSON files under tests/data/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_cases(filename: str) -> List[Dict[str, Any]]:
    """Load a list of test cases from ``tests/data/<filename>``."""
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))
