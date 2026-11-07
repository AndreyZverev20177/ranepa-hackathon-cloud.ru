"""Small shared file-IO helpers used by separate research CLIs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON result without exposing an incomplete document on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.tmp-', suffix='.json', delete=False) as stream:
            temp = stream.name
            json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'))
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if temp is not None:
            Path(temp).unlink(missing_ok=True)
