"""Independent read-only compiler-IR E0/E1 paper table exporter.

Consumes actual source-pinned create-only CSV/JSON campaign outputs; rejects
missing or inconsistent rows, and never turns generated-source checks into
claimed native synthesis or physical results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read(root, kind):
    folder = Path(root)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Missing or symlinked " + kind + " study")
    summary = folder/"summary.json"
    cases = folder/"cases.csv"
    if not summary.is_file() or not cases.is_file() or summary.is_symlink() or cases.is_symlink():
        raise ValueError(kind + " study must contain genuine summary and cases")
    info = json.loads(summary.read_text())
    with cases.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not info.get("source_revision"):
        raise ValueError(kind + " study has no source-pinned rows")
    return info, rows


def report(*, array_study: str | Path, row_study: str | Path,
           chain_study: str | Path, output: str | Path) -> dict:
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only paper analysis output exists")
    a, ar = _read(array_study, "array")
    b, br = _read(row_study, "row")
    c, cr = _read(chain_study, "chain")
    if a["schema"] != "fpgai-research/compositional-ir-cpu-campaign/v1":
        raise ValueError("Unrecognized array campaign")
    if b["schema"] != "fpgai-research/row-lifetime-cpu-campaign/v1":
        raise ValueError("Unrecognized row-lifetime campaign")
    if c["schema"] != "fpgai-research/compositional-row-chain-cpu/v1":
        raise ValueError("Unrecognized row-chain campaign")
    if len(ar) != a["cases"] or len(br) != b["paired_cases"] or len(cr) != c["cases"]:
        raise ValueError("Study summary row counts disagree with original CSV")
    sources = [a["source_revision"], b["source_revision"], c["source_revision"]]
    if len(set(sources)) != 1:
        raise ValueError("Paper table must not merge different compiler revisions")
    checks = [
        ("Array layout", a["cases"], a["host_checked_cases"],
         a["host_numeric_passes"], "Generated C++ access permutations"),
        ("Row lifetime", b["paired_cases"], b["host_checked_pairs"],
         b["host_both_qualified"], "Source scratch declarations, not BRAM"),
        ("Multi-operator chain", c["cases"], c["host_checked"],
         c["host_numeric_passes"], "3–4 operator ordered row epochs"),
    ]
    if any(checked > cases or passed > checked for _,cases,checked,passed,_ in checks):
        raise ValueError("Inconsistent numerical qualification counters")
    doc = [
        "# FPGAI CST-Model IR: source-checked CPU experiment results",
        "",
        f"**Source revision:** `{sources[0]}`",
        "",
        "| CPU experiment | Cases/pairs | Host checked | Host accepted | Evidence scope |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, cases, checked, passed, scope in checks:
        doc.append(f"| {name} | {cases} | {checked} | {passed} | {scope} |")
    doc.extend([
        "",
        f"- **Array mappings:** {a['distinct_functional_graphs']} distinct functional graphs; "
        f"{a['graphs_with_at_least_two_architecture_mappings']} graphs have at least two "
        f"distinct effective per-graph architecture mappings. Across all graphs there are "
        f"{a['distinct_effective_architectures']} combined graph+mapping signatures and "
        f"{a['distinct_generated_cpp']} generated C++ hashes. "
        "Do not infer distinct RTL or superior hardware.",
        f"- **Row lifetime:** maximum declared-source scratch ratio {b['max_source_scratch_reduction_ratio']}×. "
        "This is *not* measured FPGA RAM savings.",
        f"- **Multi-operator composition:** {c['chains_with_three_operators']} three-operator and "
        f"{c['chains_with_four_operators']} four-operator cases; "
        f"{c['phi_and_row_epoch_traces']} independently executed Phi/row-epoch traces.",
        "",
        "**Evidence E0/E1 only.** These summaries do not establish Vitis CSim, "
        "CSynth, real Fmax, RTL equivalence, or novelty priority versus prior art.",
        "",
    ])
    dest.mkdir(parents=True)
    (dest/"cpu_results.md").write_text("\n".join(doc))
    metadata = {
        "schema": "fpgai-research/cst-cpu-paper-table/v1",
        "source_revision": sources[0],
        "original_host_checked_total": sum(x[2] for x in checks),
        "original_host_numerical_passes": sum(x[3] for x in checks),
        "no_vitis_source_evidence_promoted": True,
        "prior_art_novelty_established": False,
        "report_is_read_only_on_input_studies": True,
    }
    (dest/"report.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array-study", required=True)
    p.add_argument("--row-study", required=True)
    p.add_argument("--chain-study", required=True)
    p.add_argument("--output", required=True)
    x = p.parse_args(argv)
    print(json.dumps(report(array_study=x.array_study, row_study=x.row_study,
                            chain_study=x.chain_study, output=x.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
