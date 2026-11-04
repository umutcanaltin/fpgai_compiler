"""Inspect original E0-E4 receipts before ranking one comparable paper campaign."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from fpgai_research.experiments.pareto import analyze_pareto, DEFAULT_OBJECTIVES
from fpgai_research.experiments.paper_tables import make_rows
from fpgai_research.experiments.results import summarize_case

def analyze_campaign(directory: str | Path, destination: str | Path,
                     objectives: tuple[str,...] = DEFAULT_OBJECTIVES) -> dict:
    root=Path(directory)
    index=root/"campaign.json"
    campaign=json.loads(index.read_text())
    if campaign.get("schema")!="fpgai-research/paper-campaign/v1":
        raise ValueError("Not a prepared paper campaign")
    ready=[r for r in campaign.get("cases",()) if r.get("status")=="READY"]
    if not ready:
        raise ValueError("Pareto requires prepared model cases")
    seen=set()
    reports=[]
    provenance_key=None
    for item in ready:
        name=item["case"]
        if not isinstance(name,str) or not name.startswith("case_") or not name[5:].isdigit() or name in seen:
            raise ValueError("Unsafe or duplicate case")
        seen.add(name)
        case=root/name
        report=summarize_case(str(case))
        if report["source_sha256"]!=item.get("cpp_sha256"):
            raise ValueError("Index/source hash mismatch")
        manifest=json.loads((case/"manifest.json").read_text())
        if manifest.get("part")!=campaign.get("part") or manifest.get("clock_ns")!=campaign.get("clock_ns"):
            raise ValueError("Target/clock mismatch across cases")
        provenance=json.loads((case/"inputs_provenance.json").read_text())
        sources=provenance.get("source_provenance")
        if not isinstance(sources,dict) or not all(sources.get(k) for k in ("onnx_sha256","inputs_npy_sha256")):
            raise ValueError("Missing model and input source hashes")
        key=(sources["onnx_sha256"],sources["inputs_npy_sha256"],sources.get("labels_npy_sha256"))
        if provenance_key is None:
            provenance_key=key
        elif key!=provenance_key:
            raise ValueError("Cases have different model, inputs or labels")
        reports.append(report)
    table_rows=make_rows(campaign,{"schema":"fpgai-research/paper-evidence/v2","results":reports})
    result=analyze_pareto(table_rows,objectives)
    result.update(campaign_sha256=sha256(index.read_bytes()).hexdigest(),
                  model_sha256=provenance_key[0],inputs_sha256=provenance_key[1],
                  labels_sha256=provenance_key[2],
                  part=campaign["part"],clock_ns=campaign["clock_ns"],
                  case_evidence_collected_from_original_receipts=True)
    output=Path(destination)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result
