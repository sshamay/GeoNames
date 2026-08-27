"""Load JSON test-case files for the AQuA framework.

Portable: the data directory is explicit (argument) or read from the
``AQUA_DATA_DIR`` env var, defaulting to ``tests/data`` relative to the
working directory (where pytest runs). Host projects can point it anywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_DATA_DIR = "tests/data"


def load_cases(filename: str, data_dir: str = None) -> List[Dict[str, Any]]:
    """Load a list of test cases from a JSON file under ``data_dir``.

    ``data_dir`` defaults to ``$AQUA_DATA_DIR`` or ``tests/data`` (relative to
    the working directory).
    """
    base = data_dir or os.environ.get("AQUA_DATA_DIR") or DEFAULT_DATA_DIR
    return json.loads((Path(base) / filename).read_text(encoding="utf-8"))
