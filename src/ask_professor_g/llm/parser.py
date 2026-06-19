from __future__ import annotations

import json
import re
from typing import Any


_FENCED_BLOCK = re.compile(r"```(?:json|python)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _strip_fence(text: str) -> str:
    match = _FENCED_BLOCK.search(text.strip())
    return match.group(1).strip() if match else text.strip()


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    candidate = _strip_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start_obj = candidate.find("{")
        start_arr = candidate.find("[")
        starts = [idx for idx in [start_obj, start_arr] if idx >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end <= start:
            raise
        return json.loads(candidate[start : end + 1])


def extract_python(text: str) -> str:
    candidate = _strip_fence(text)
    lines = candidate.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "#", '"""')):
            return sanitize_python("\n".join(lines[idx:]).strip())
    return sanitize_python(candidate.strip())


def sanitize_python(code: str) -> str:
    code = code.strip()
    if ("np." in code or "np.ndarray" in code) and "import numpy as np" not in code:
        code = "import numpy as np\n\n" + code
    return code
