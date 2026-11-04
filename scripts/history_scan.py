"""Conservative, bounded Git history audit for public research releases.

Checks all reachable commits (including branches/tags) for historical
restricted filenames and basic credential signatures in every reachable
blob. A pass is not an IP/legal review, and cannot see deleted/unreachable
objects, forks, reflogs or data outside Git.
"""
from __future__ import annotations

from pathlib import Path
import io
import re
import subprocess

RESTRICTED_NAMES = frozenset({".env", ".env.local", "id_rsa", "id_ed25519",
                              "release-approval.json"})
RESTRICTED_SUFFIXES = frozenset({".onnx", ".npy", ".npz", ".pt", ".pth",
                                 ".h5", ".hdf5", ".pem", ".key",
                                 ".p12", ".pfx", ".bit", ".hwh"})
SECRET_PATTERNS = (
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)

def _git(root: Path, args: list[str], *, input_data: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=root, input=input_data,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Git history inspection unavailable") from exc
    if result.returncode:
        raise ValueError(f"Git history inspection failed: {args[0]}")
    return result.stdout

def scan_history(root: Path, *, max_objects=50000,
                 max_blob_bytes=2_000_000, max_total_bytes=150_000_000) -> dict:
    """Exhaustively inspect reachable blobs or fail closed on size/resource bounds."""
    root = root.resolve(strict=True)
    # A shallow clone cannot establish that deleted secrets are absent
    # from the repository's earlier, un-fetched history.
    shallow = _git(root, ["rev-parse", "--is-shallow-repository"]).strip()
    if shallow != b"false":
        raise ValueError("Shallow or indeterminate Git history is unsuitable for public release")
    listing = _git(root, ["rev-list", "--objects", "--all"])
    parsed = [line.split(b" ", 1) for line in listing.splitlines() if line]
    if not parsed:
        raise ValueError("No committed Git history to audit")
    if len(parsed) > max_objects:
        raise ValueError("Git object count exceeds conservative audit budget")
    objects = [item[0] for item in parsed]
    risks = []
    for obj in parsed:
        if len(obj) < 2:
            continue
        name = obj[1].decode("utf-8", errors="replace")
        basename = Path(name).name
        if basename in RESTRICTED_NAMES or Path(name).suffix.lower() in RESTRICTED_SUFFIXES:
            risks.append(f"Restricted historical artifact: {name}")
    batch = b"\n".join(objects) + b"\n"
    headers = _git(root, ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                   input_data=batch).splitlines()
    if len(headers) != len(objects):
        raise ValueError("Incomplete Git object inventory")
    blobs = []
    sizes = {}
    for header in headers:
        fields = header.split()
        if len(fields) != 3:
            raise ValueError("Malformed Git object metadata")
        sha, kind, raw_size = fields
        if kind != b"blob":
            continue
        size = int(raw_size)
        if size > max_blob_bytes:
            risks.append(f"Historical blob exceeds manual-review size cap: {sha.decode('ascii')}")
            continue
        blobs.append(sha)
        sizes[sha] = size
    if sum(sizes.values()) > max_total_bytes:
        raise ValueError("Historical blob scan exceeds conservative total byte budget")
    if blobs:
        raw = _git(root, ["cat-file", "--batch"], input_data=b"\n".join(blobs) + b"\n")
        stream = io.BytesIO(raw)
        for expected in blobs:
            parts = stream.readline().strip().split()
            if len(parts) != 3 or parts[0] != expected or parts[1] != b"blob":
                raise ValueError("Malformed Git history blob framing")
            size = int(parts[2])
            if size != sizes[expected]:
                raise ValueError("Git object size changed during history scan")
            content = stream.read(size)
            if len(content) != size or stream.read(1) != b"\n":
                raise ValueError("Truncated Git history blob")
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                risks.append(f"Potential historical secret in Git blob: {expected.decode('ascii')}")
        if stream.read(1):
            raise ValueError("Unexpected trailing Git object bytes")
    return {"reachable_objects_examined": len(objects),
            "reachable_blobs_scanned": len(blobs),
            "historical_risks": sorted(set(risks)),
            "history_scan_passed": not risks,
            "limitations": "Does not review unreachable objects, forks, external artifacts, private keys that do not match patterns, copyright/IP rights or commit authors"}
