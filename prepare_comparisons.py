"""Install required comparison files from an external dependency ZIP."""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "comparisons" / "common"))
from dependencies import PACKAGE_ROOT, RECEIPT_NAME, cache_root, required_files, verify_files


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_argument(value):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("--sha256 must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="local dependency ZIP outside this package")
    source.add_argument("--archive-url", help="public HTTPS URL for an anonymous dependency ZIP")
    source.add_argument("--verify-only", action="store_true", help="check the prepared external cache offline")
    parser.add_argument("--sha256", type=sha256_argument, help="expected ZIP SHA-256; required with --archive-url")
    parser.add_argument("--cache-root", "--cache-dir", dest="cache_dir", type=Path, help="external cache directory")
    args = parser.parse_args()

    if args.archive_url:
        url = urlsplit(args.archive_url)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.fragment:
            parser.error("--archive-url must be a public HTTPS URL without credentials or a fragment")
        if args.sha256 is None:
            parser.error("--sha256 is required with --archive-url")
    if args.archive:
        archive = args.archive.expanduser().resolve()
        if archive == PACKAGE_ROOT or PACKAGE_ROOT in archive.parents:
            parser.error("--archive must be outside this package")
        if not archive.is_file():
            parser.error("--archive must name an existing ZIP file")

    root = cache_root(args.cache_dir)
    manifest = required_files()
    files = sorted({relative for method_files in manifest.values() for relative in method_files})
    if args.verify_only:
        for method in manifest:
            verify_files(method, root)
        receipt = json.loads((root / RECEIPT_NAME).read_text())
        if args.sha256 and receipt["archive_sha256"] != args.sha256:
            raise ValueError("cached dependency archive SHA-256 mismatch")
        print(json.dumps({"verified_files": len(files), "archive_sha256": receipt["archive_sha256"]}, indent=2))
        return

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
        staging = Path(temporary)
        if args.archive_url:
            archive = staging / "dependencies.zip"
            request = Request(args.archive_url, headers={"User-Agent": "anonymous-comparison-setup"})
            with urlopen(request, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)

        archive_sha256 = sha256_file(archive)
        if args.sha256 and archive_sha256 != args.sha256:
            raise ValueError("dependency archive SHA-256 mismatch")

        digests = {}
        with ZipFile(archive) as bundle:
            entries = {}
            for info in bundle.infolist():
                entries.setdefault(info.filename, []).append(info)
            for relative in files:
                candidates = entries.get(f"code/{relative}", []) + entries.get(relative, [])
                if len(candidates) != 1 or candidates[0].is_dir():
                    raise FileNotFoundError(f"archive requires exactly one regular file: {relative}")
                info = candidates[0]
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"archive dependency must not be a symlink: {relative}")
                payload = bundle.read(info)
                target = staging / "code" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                digests[relative] = hashlib.sha256(payload).hexdigest()

        for relative in files:
            target = root / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / "code" / relative, target)
        receipt = {"format": 1, "archive_sha256": archive_sha256, "files": digests}
        (staging / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(staging / RECEIPT_NAME, root / RECEIPT_NAME)

    print(json.dumps({"installed_files": len(files), "archive_sha256": archive_sha256}, indent=2))


if __name__ == "__main__":
    main()
