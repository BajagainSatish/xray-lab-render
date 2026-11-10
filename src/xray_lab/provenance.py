"""Local artifact and source records; no remote tracking service."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def code_record(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    try:
        git_record = {"commit": git("rev-parse", "HEAD"),
                      "status_porcelain": git("status", "--porcelain")}
    except (OSError, subprocess.CalledProcessError) as exc:
        git_record = {"unavailable": str(exc)}
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += sorted(root.glob("requirements*.txt"))
    return {"git": git_record,
            "file_sha256": {str(p.relative_to(root)): sha256_file(p) for p in paths}}
