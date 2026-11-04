"""Combine scientific Model IR evidence and attested native metrics across models.

NumPy E0 accuracy, E1 source/GCC findings and E2-E4 native results remain
distinct. Unmeasured values are null in JSON and TBD in CSV.
"""
from __future__ import annotations
from hashlib import sha256
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.paper_tables import COLUMNS, make_rows
from fpgai_research.experiments.results import summarize_case

EXTRAS=("model","family","functional_sha256","reference_top1_e0",
        "model_ir_changed_operators","lowered_ir_changed_operators")

def collect_suite_tables(directory,output_dir,*,require_host=False):
    """Write complete but never fabricated suite paper-result tables."""
    if type(require_host) is not bool:
        raise ValueError("Host evidence requirement must be boolean")
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Suite must be a regular source directory")
    audit=audit_suite(root,require_host=require_host,require_all_axes=False)
    declaration=json.loads((root/"suite_results.json").read_text())
    studies=declaration["studies"]
    if len(studies)!=audit["requested_models"]:
        raise ValueError("Suite model inventory inconsistent")
    rows=[]
    for study in studies:
        name=study["id"]
        family=study["family"]
        if study["status"] in ("BLOCKED_ASSETS","BLOCKED_UNSUPPORTED","BLOCKED_VALIDATION"):
            item={key:None for key in COLUMNS}
            item.update(model=name,family=family,status=study["status"],
                        blocker=study.get("reason"),functional_sha256=None,
                        reference_top1_e0=None,model_ir_changed_operators=None,
                        lowered_ir_changed_operators=None)
            rows.append(item)
            continue
        target=root/name
        campaign=json.loads((target/"campaign.json").read_text())
        scientific=json.loads((target/"full_experiment.json").read_text())
        representation=json.loads((target/"representation_ablation.json").read_text())
        mapping={entry["case"]:entry for entry in representation["cases"]}
        results=[summarize_case(str(target/row["case"])) for row in campaign["cases"]
                 if row["status"]=="READY"]
        collected=make_rows(campaign,{"schema":"fpgai-research/paper-evidence/v2","results":results})
        native_by_case={entry["name"]:entry for entry in results}
        reference=scientific["reference_evaluation"]
        if reference.get("schema")!="fpgai-research/reference-evaluation/v1":
            raise ValueError("Unexpected E0 numerical reference evidence")
        for cell in collected:
            ablation=mapping[cell["case"]]
            if ablation["status"]=="PASS":
                if ablation["cpp_sha256"]!=native_by_case[cell["case"]]["source_sha256"]:
                    raise ValueError("Scientific Model IR report does not match verified generated source")
            cell.update(model=name,family=family,
                functional_sha256=scientific["functional_sha256"],
                reference_top1_e0=reference["accuracy_top1"],
                model_ir_changed_operators=ablation.get("changed_architectural_operators"),
                lowered_ir_changed_operators=ablation.get("changed_lowered_operators"))
            rows.append(cell)
    destination=Path(output_dir)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Cannot replace existing suite paper results")
    evidence_counts={}
    for row in rows:
        label=row.get("evidence") or row.get("status") or "UNAVAILABLE"
        evidence_counts[label]=evidence_counts.get(label,0)+1
    manifest={"schema":"fpgai-research/full-paper-suite-tables/v1",
        "suite_sha256":sha256((root/"suite_results.json").read_bytes()).hexdigest(),
        "model_count":audit["requested_models"],"case_count":len(rows),
        "evidence_counts":evidence_counts,
        "all_models_source_ready":audit["all_models_source_ready"],
        "all_native_paper_experiments_complete":audit["all_native_paper_experiments_complete"],
        "limitations":"E0 classification and E1 GCC validation are not native fixed-point or routed FPGA performance",
        "rows":rows}
    columns=EXTRAS+COLUMNS
    with TemporaryDirectory(prefix=".fpgai-suite-tables-",dir=destination.parent) as folder:
        stage=Path(folder)
        (stage/"suite_results.json").write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
        with (stage/"suite_results.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writer.writerows({field:("TBD" if row.get(field) is None else
                 ";".join(row[field]) if isinstance(row.get(field),list)
                 else row[field]) for field in columns} for row in rows)
        stage.rename(destination)
    return manifest
