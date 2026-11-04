"""Immutable, transportable input archive for model-only paper reproduction.

Captures exact research inputs alongside generated cases, with SHA-256
verification and an optional external provenance cross-check. This is an
integrity aid (not a signature, encryption, license or vendor measurement).
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

SCHEMA = "fpgai-research/input-archive/v1"
FILES = {
    "onnx": "model.onnx",
    "samples": "samples.npy",
    "labels": "labels.npy",
    "architecture": "architecture.json",
    "plan": "campaign_plan.json",
    "metadata": "training_metadata.json",
}
PROVENANCE_KEYS = {
    "onnx": "onnx_sha256",
    "samples": "inputs_npy_sha256",
    "labels": "labels_npy_sha256",
    "architecture": "architecture_sha256",
    "plan": "plan_sha256",
    "metadata": "metadata_sha256",
}


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Input archive requires a regular source file: {candidate}")
    return candidate.resolve(strict=True)


def archive_inputs(directory: str | Path, *, model: str | Path,
                   samples: str | Path, labels: str | Path | None = None,
                   architecture: str | Path | None = None,
                   plan: str | Path | None = None,
                   metadata: str | Path | None = None) -> dict:
    """Atomically add inputs/ to an existing case or campaign directory.

    This is explicit because ONNX weights and training metadata may be private.
    Existing archives are never changed, including on failed commands.
    """
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    target = root / "inputs"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Research input archive already exists: {target}")
    requested = {"onnx": model, "samples": samples, "labels": labels,
                 "architecture": architecture, "plan": plan, "metadata": metadata}
    sources = {role: _source(path) for role, path in requested.items() if path is not None}
    temp = Path(tempfile.mkdtemp(prefix=".fpgai-inputs-", dir=root))
    try:
        records = {}
        for role, source in sorted(sources.items()):
            filename = FILES[role]
            destination = temp / filename
            with source.open("rb") as inp, destination.open("xb") as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
            if _digest(destination) != _digest(source):
                raise ValueError(f"Source changed during archival: {role}")
            records[role] = {"file": filename, "bytes": destination.stat().st_size,
                             "sha256": _digest(destination)}
        manifest = {"schema": SCHEMA,
                    "privacy": "May contain proprietary model weights and personal dataset metadata",
                    "entries": records,
                    "note": "Content hashes attest stored bytes, not model accuracy, origin or legal ownership"}
        (temp / "archive.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        if target.exists():
            raise FileExistsError(target)
        temp.rename(target)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def verify_archive(directory: str | Path, *,
                   expected_sha256: Mapping[str, str] | None = None) -> dict:
    """Check exact archived file inventory, sizes and hashes; fail closed."""
    folder = Path(directory) / "inputs"
    if folder.is_symlink() or not folder.is_dir():
        raise FileNotFoundError(f"No regular input archive: {folder}")
    meta = folder / "archive.json"
    if meta.is_symlink() or not meta.is_file():
        raise FileNotFoundError("Input archive manifest missing")
    record = json.loads(meta.read_text())
    if record.get("schema") != SCHEMA or not isinstance(record.get("entries"), dict):
        raise ValueError("Unsupported input archive schema")
    entries = record["entries"]
    if not {"onnx", "samples"} <= set(entries) or set(entries) - set(FILES):
        raise ValueError("Archive must contain model and samples, with only recognized roles")
    listed = {"archive.json"}
    for role, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"file", "bytes", "sha256"}:
            raise ValueError(f"Invalid input archive record: {role}")
        if entry["file"] != FILES[role]:
            raise ValueError("Unsafe or mismatched archive filename")
        content = folder / entry["file"]
        if content.is_symlink() or not content.is_file():
            raise ValueError(f"Missing or symlinked input archive file: {role}")
        if type(entry["bytes"]) is not int or content.stat().st_size != entry["bytes"]:
            raise ValueError(f"Archived input size mismatch: {role}")
        if _digest(content) != entry["sha256"]:
            raise ValueError(f"Archived input SHA-256 mismatch: {role}")
        listed.add(entry["file"])
    if {p.name for p in folder.iterdir()} != listed:
        raise ValueError("Unlisted archive content found")
    if expected_sha256:
        for role, digest in expected_sha256.items():
            if role not in entries or entries[role]["sha256"] != digest:
                raise ValueError(f"Archived {role} differs from external provenance")
    return {"schema": "fpgai-research/input-archive-verification/v1",
            "integrity_verified": True, "roles": sorted(entries),
            "archived_bytes": sum(e["bytes"] for e in entries.values()),
            "native_tools_executed": False,
            "limitations": "Content integrity only; no signature or external provenance unless expected hashes supplied"}


def verify_case_provenance(directory: str | Path) -> dict:
    """Bind case/campaign archive bytes to hashes recorded at HLS preparation."""
    root = Path(directory)
    case = root / "inputs_provenance.json"
    campaign = root / "campaign.json"
    if case.is_file():
        record = json.loads(case.read_text())
        expected = record.get("source_provenance", record)
    elif campaign.is_file():
        record = json.loads(campaign.read_text())
        if record.get("schema") != "fpgai-research/paper-campaign/v1":
            raise ValueError("Expected prepared paper campaign")
        ready = [item for item in record.get("cases", ()) if item.get("status") == "READY"]
        if not ready:
            raise ValueError("Campaign contains no READY cases to bind inputs")
        payloads = []
        for item in ready:
            name = item.get("case", "")
            if not isinstance(name, str) or not name.startswith("case_") or not name[5:].isdigit():
                raise ValueError("Unsafe campaign case identity")
            candidate = root / name / "inputs_provenance.json"
            if not candidate.is_file():
                raise ValueError("Campaign case lacks input provenance: " + name)
            body = json.loads(candidate.read_text())
            payloads.append(body.get("source_provenance"))
        expected = payloads[0]
        if any(value != expected for value in payloads):
            raise ValueError("Campaign contains conflicting source provenance")
    else:
        raise FileNotFoundError("Missing prepared input provenance")
    if not isinstance(expected, dict):
        raise ValueError("Invalid input provenance")
    hashes = {role: expected[key] for role, key in PROVENANCE_KEYS.items()
              if expected.get(key) is not None}
    if not {"onnx", "samples"} <= set(hashes):
        raise ValueError("Source model/input hashes missing from preparation receipt")
    return verify_archive(root, expected_sha256=hashes)
