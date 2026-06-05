from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""
    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    if not env_path.exists():
        package_root_env = Path(__file__).resolve().parent.parent / ".env"
        env_path = package_root_env
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value
