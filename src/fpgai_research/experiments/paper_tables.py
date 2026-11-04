"""Paper CSV/Markdown export; unknown data is TBD, HLS and P&R are separate."""
from __future__ import annotations
from pathlib import Path
import csv
import json

COLUMNS=("case","axis","status","blocker","evidence","precision_bits",
         "integer_bits","fractional_bits","numerical_tolerance",
         "csim_failures","cosine","rmse","max_abs_error","numerical_pass",
         "reference_top1","compiled_top1","class_disagreement_rate","labeled_samples",
         "hls_latency_best","hls_latency_worst","hls_ii_min",
         "hls_lut_estimate","hls_ff_estimate","hls_dsp_estimate",
         "hls_bram18_estimate","hls_uram_estimate",
         "postroute_lut","postroute_ff","postroute_dsp","postroute_bram",
         "postroute_uram","postroute_wns_ns","postroute_timing_met",
         "postroute_drc_violations","postroute_drc_critical_warnings",
         "postroute_io_unconstrained","postroute_implementation_mode")


def make_rows(campaign: dict, evidence: dict) -> list[dict]:
    if campaign.get("schema")!="fpgai-research/paper-campaign/v1":
        raise ValueError("Expected paper campaign v1")
    if evidence.get("schema")!="fpgai-research/paper-evidence/v2":
        raise ValueError("Expected attested paper evidence v2")
    records={r["name"]:r for r in evidence.get("results",[])}
    selected={r["case"] for r in campaign["cases"] if r["status"]=="READY"}
    if set(records)-selected or len(records)!=len(evidence.get("results",[])):
        raise ValueError("Unexpected or repeated result case IDs")
    rows=[]
    for item in campaign["cases"]:
        r=records.get(item["case"])
        status=item["status"]
        if status=="READY" and r is None:
            raise ValueError("Missing collected evidence for prepared case: "+item["case"])
        if status=="BLOCKED" and r is not None:
            raise ValueError("Blocked case cannot have measured evidence")
        row={k:None for k in COLUMNS}
        bits=item["requested"]["precision_bits"]
        integer=item["requested"].get("integer_bits")
        row.update(case=item["case"],axis=item["dimension"],status=status,
                   blocker=(item.get("reason") or {}).get("code"),
                   precision_bits=bits,integer_bits=integer,
                   fractional_bits=(bits-integer
                                    if type(bits) is int and type(integer) is int else None))
        if r is not None:
            row["evidence"]=r["evidence"]
            csim=r.get("csim_receipt") or {}
            row["numerical_tolerance"]=r.get("numerical_tolerance")
            if csim.get("failures") is not None:
                row["csim_failures"]=csim["failures"]
            if r.get("csim_passed") is not None:
                row.update(cosine=csim.get("cosine"),rmse=csim.get("rmse"),
                           max_abs_error=csim.get("max_absolute_error"),
                           numerical_pass=r["csim_passed"])
            cls=csim.get("classification") or {}
            if r.get("csim_receipt") and cls:
                row.update(reference_top1=cls.get("reference_top1_accuracy"),
                           compiled_top1=cls.get("compiled_top1_accuracy"),
                           class_disagreement_rate=cls.get("class_disagreement_rate"),
                           labeled_samples=cls.get("samples"))
            if r["evidence"] in ("E3-HLS-synthesis-report","E4-Vivado-implementation"):
                hls=r.get("metrics") or {}
                for old,new in (("latency_best","hls_latency_best"),
                                ("latency_worst","hls_latency_worst"),("ii_min","hls_ii_min"),
                                ("lut","hls_lut_estimate"),("ff","hls_ff_estimate"),
                                ("dsp","hls_dsp_estimate"),("bram_18k","hls_bram18_estimate"),
                                ("uram","hls_uram_estimate")):
                    row[new]=hls.get(old)
            if r["evidence"]=="E4-Vivado-implementation":
                physical=r.get("vivado_metrics") or {}
                usage=physical.get("utilization") or {}
                for old,new in (("lut","postroute_lut"),("ff","postroute_ff"),
                                ("dsp","postroute_dsp"),("bram","postroute_bram"),
                                ("uram","postroute_uram")):
                    row[new]=usage.get(old)
                timing=physical.get("timing") or {}
                row["postroute_wns_ns"]=timing.get("wns_ns")
                row["postroute_timing_met"]=timing.get("timing_met")
                row["postroute_implementation_mode"]=r.get("vivado_implementation_mode")
                drc=physical.get("drc") or {}
                if drc.get("summary_verified") is True:
                    row["postroute_drc_violations"]=drc.get("violations")
                    row["postroute_drc_critical_warnings"]=drc.get("critical_warnings")
                    row["postroute_io_unconstrained"]=drc.get("unconstrained_io")
        rows.append(row)
    return rows


def export_tables(campaign_path: str | Path, evidence_path: str | Path,
                  output_dir: str | Path) -> list[dict]:
    rows=make_rows(json.loads(Path(campaign_path).read_text()),
                   json.loads(Path(evidence_path).read_text()))
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    def show(value):return "TBD" if value is None else str(value)
    with (root/"paper_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:show(row[k]) for k in COLUMNS})
    with (root/"paper_results.md").open("w") as f:
        f.write("| "+" | ".join(COLUMNS)+" |\n")
        f.write("| "+" | ".join("---" for _ in COLUMNS)+" |\n")
        for row in rows:
            f.write("| "+" | ".join(show(row[k]) for k in COLUMNS)+" |\n")
        f.write("\nTBD: no attested evidence at this level. "
                "HLS estimates are distinct from post-route utilization.\n")
    return rows
