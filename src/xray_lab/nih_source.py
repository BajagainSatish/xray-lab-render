"""Pinned access to a public NIH mirror that exposes individual release files."""

from pathlib import Path
import hashlib
import json
import re
import tempfile

import requests

from xray_lab.provenance import sha256_file, utc_now

MIRROR_REPO = "iraqigold/nih-chest-xray-14-flat"
MIRROR_REVISION = "98fdac982847ed878cb68065d9e9f66998a79689"
MIRROR_PAGE = f"https://huggingface.co/datasets/{MIRROR_REPO}"
MIRROR_FOLDER = "images_001/images"
SECOND_MIRROR = "yeigen/nih-chest-xray"
PUBLISHER_PAGE = "https://nihcc.app.box.com/v/ChestXray-NIHCC"
DATASET_DOC = (
    "https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest"
)

# SHA256 was identical in two independent mirrors of the official release.
EXPECTED_FILES = {
    "Data_Entry_2017.csv": (
        "88f75094e25ccc0c6f1f9cdfd4b2f94f9379a0ae07d5ff4dcf94242707b07462",
        7_861_152,
    ),
    "test_list.txt": (
        "38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc",
        435_131,
    ),
}
MAX_LABEL_FILE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def resolve_url(path: str) -> str:
    return f"{MIRROR_PAGE}/resolve/{MIRROR_REVISION}/{path}"


def download_verified(url: str, target: Path, expected_sha: str, expected_size: int,
                      max_bytes: int) -> str:
    if target.is_file():
        if target.stat().st_size != expected_size or sha256_file(target) != expected_sha:
            raise RuntimeError(f"Cached file differs from its pinned hash: {target}")
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            with requests.get(url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(256 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise RuntimeError(f"Download exceeds the {max_bytes} byte limit: {url}")
                    digest.update(chunk)
                    output.write(chunk)
        if size != expected_size:
            raise RuntimeError(f"Size mismatch for {target.name}: {size} != {expected_size}")
        if digest.hexdigest() != expected_sha:
            raise RuntimeError(f"SHA256 mismatch for {target.name}; do not use this file.")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "downloaded"


def mirror_listing(folder: str) -> dict[str, dict]:
    """Page the mirror's file API and return name -> size/lfs hash/path."""
    url = f"https://huggingface.co/api/datasets/{MIRROR_REPO}/tree/main/{folder.replace('/', '%2F')}"
    listing: dict[str, dict] = {}
    while url:
        response = requests.get(url, timeout=(10, 60))
        response.raise_for_status()
        for item in response.json():
            if item["type"] != "file":
                continue
            name = item["path"].rsplit("/", 1)[-1]
            listing[name] = {
                "path": item["path"],
                "size": item["size"],
                "sha256": item["lfs"]["oid"],
            }
        match = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("link", ""))
        url = match.group(1) if match else None
    if not listing:
        raise RuntimeError(f"Mirror folder is empty: {folder}")
    return listing


IMAGE_FOLDERS = tuple(f"images_{index:03d}/images" for index in range(1, 13))


def mirror_listing_all(cache_path: Path | None = None) -> dict[str, dict]:
    """Index every NIH image folder once and cache the result."""
    if cache_path and cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if (cached.get("revision") == MIRROR_REVISION
                and len(cached.get("files", {})) > 0):
            return cached["files"]
    files: dict[str, dict] = {}
    for folder in IMAGE_FOLDERS:
        for name, record in mirror_listing(folder).items():
            if name in files:
                raise RuntimeError(f"Duplicate image name across folders: {name}")
            files[name] = record
        print(f"  indexed {folder}: {len(files)} total", flush=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "revision": MIRROR_REVISION,
            "updated_at_utc": utc_now(),
            "files": files,
        }))
    return files
