"""Verify method-specific files in the external comparison cache."""

import hashlib
import json
from pathlib import Path


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "environment-model" / "comparisons"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "external_files.json"
RECEIPT_NAME = "receipt.json"


def cache_root(path=None):
    root = (Path(path).expanduser() if path is not None else DEFAULT_CACHE_ROOT).resolve()
    if root == PACKAGE_ROOT or PACKAGE_ROOT in root.parents:
        raise ValueError("dependency cache must be outside the anonymous package")
    return root


def external_code_root(path=None):
    return cache_root(path) / "code"


def required_files():
    return json.loads(MANIFEST_PATH.read_text())


def verify_files(method, path=None):
    expected = required_files()[method]
    root = cache_root(path)
    receipt_path = root / RECEIPT_NAME
    if not receipt_path.is_file():
        raise FileNotFoundError("missing external dependency receipt")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != 1 or not isinstance(receipt.get("files"), dict):
        raise ValueError("invalid external dependency receipt")
    code = root / "code"
    for relative in expected:
        digest = receipt["files"].get(relative)
        file = code / relative
        if not digest or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise FileNotFoundError(f"missing or mismatched fixed dependency: {relative}")
    return code
