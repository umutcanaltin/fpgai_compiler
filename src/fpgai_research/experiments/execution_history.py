"""Append-only invocation history; vendor command completion is not FPGA evidence."""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

def record_execution(root: str | Path, kind: str, receipt: dict) -> Path:
    if kind not in ("campaign","suite") or not isinstance(receipt,dict):
        raise ValueError("Only canonical campaign or suite execution receipts are supported")
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Execution history requires a regular study directory")
    folder=root/"execution_history"
    if folder.is_symlink():
        raise ValueError("Execution history cannot be a symlink")
    folder.mkdir(exist_ok=True)
    if not folder.is_dir():
        raise ValueError("Execution history path is not a directory")
    canonical=json.dumps(receipt,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    envelope={
        "schema":"fpgai-research/append-only-execution-receipt/v1",
        "kind":kind,
        "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
        "receipt_sha256":sha256(canonical).hexdigest(),
        "receipt":receipt,
        "disclaimer":"Command outcome only; E2/E3/E4 require independent attested native evidence"}
    destination=folder/f"{kind}-{uuid4().hex}.json"
    with destination.open("x",encoding="utf-8") as stream:
        json.dump(envelope,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    return destination

def verify_execution_history(root: str | Path) -> dict:
    folder=Path(root)/"execution_history"
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Missing or linked execution history directory")
    entries=[]
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix!=".json":
            raise ValueError("Unexpected file in execution history")
        record=json.loads(path.read_text())
        if record.get("schema")!="fpgai-research/append-only-execution-receipt/v1":
            raise ValueError("Unknown execution history schema")
        kind=record.get("kind")
        if kind not in ("campaign","suite") or not path.name.startswith(kind+"-"):
            raise ValueError("Invalid history record kind")
        canonical=json.dumps(record.get("receipt"),sort_keys=True,
                             separators=(",",":"),allow_nan=False).encode("utf-8")
        if sha256(canonical).hexdigest()!=record.get("receipt_sha256"):
            raise ValueError("Execution receipt content hash mismatch")
        entries.append({"file":path.name,"kind":kind,
                        "receipt_sha256":record["receipt_sha256"]})
    return {"schema":"fpgai-research/execution-history-audit/v1",
            "receipt_count":len(entries),"entries":entries,
            "evidence":"Invocation history, not native measurement"}
