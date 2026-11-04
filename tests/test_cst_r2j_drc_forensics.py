"""R2k: original DRC report identities and source attribution; zero vendor calls."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments.cst_r2j_drc_forensics import (
    SCHEMA, forensic_report, main, parse_drc_report,
)

RULES = {"DPIP-2": 3, "DPOP-3": 3, "DPOP-4": 3, "RTSTAT-10": 1}


def _raw_drc():
    blocks = []
    for rule, count in RULES.items():
        for number in range(1, count + 1):
            lines = [f"{rule}#{number} Warning"]
            if rule == "RTSTAT-10":
                lines += ["No routable loads", "26 net(s) have no routable loads.",
                          "The problem bus(es) and/or net(s) are ap_idle,"]
            else:
                lines += ["Pipelining DSP", "DSP grp_test/" + rule +
                          f"_U{number}/DSP48 input grp_test/C[47:0] is not pipelined."]
            lines += ["Related violations: <none>"]
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


@pytest.fixture
def physical_fixture(tmp_path):
    root = tmp_path / "immutable_physical"
    export = tmp_path / "immutable_export"
    root.mkdir()
    export.mkdir()
    rows = []
    pairs = []
    raw = _raw_drc().encode()
    digest = sha256(raw).hexdigest()
    for pair in range(36):
        names = [f"case_{pair*2:03d}", f"case_{pair*2+1:03d}"]
        for j, name in enumerate(names):
            observed = pair == 20
            if observed:
                path = root / name / "vivado" / "reports"
                path.mkdir(parents=True)
                (path / "drc.rpt").write_bytes(raw)
                (path.parent / "vivado_invocation.json").write_text(json.dumps({
                    "completed_successfully": True,
                    "fresh_report_sha256": {"drc.rpt": digest},
                }))
            rows.append({
                "case": name, "group": pair,
                "source_storage": "disjoint" if j == 0 else "per_element_optimal",
                "graph": f"fanout_{pair}", "mapping": "scalar",
                "source_cpp_sha256": ("a" if j == 0 else "b") * 64,
                "original_hls_qualified": True,
                "original_vivado_invocation_present": observed,
                "routed_observation_valid": observed,
                "routed_qualified": False,
                "routed_warning_observation": observed,
                "routed_timing_met": True if observed else None,
                "routed_wns_ns": (26.356 if j == 0 else 27.304) if observed else None,
                "physical_resources": ({
                    "lut": 649 if j == 0 else 684,
                    "ff": 396 if j == 0 else 407, "dsp": 7,
                    "bram": 1.5, "uram": 0,
                } if observed else None),
                "original_routed_report_sha256": (
                    {"drc.rpt": digest} if observed else None),
                "drc_warning_summary": ({
                    "summary_verified": True, "clean": False,
                    "critical_warnings": 0, "unconstrained_io": False,
                    "violations": 10,
                    "rules": {rule: {"severity": "Warning", "violations": n}
                              for rule, n in RULES.items()},
                } if observed else None),
                "drc_verified_clean": False if observed else None,
                "rom_input_evidence_mode": (
                    "ORIGINAL_SOURCE_AND_WORKING_ROM_STABLE" if observed else None),
                "requested_clock_ns": 32.0,
            })
        pairs.append({
            "pair": pair, "cases": names,
            "both_routed_observation_valid": pair == 20,
            "both_source_attributed_routed_timing_drc_qualified": False,
            "descriptive_bram_tile_difference": 0 if pair == 20 else None,
            "descriptive_lut_difference": -35 if pair == 20 else None,
            "descriptive_ff_difference": -11 if pair == 20 else None,
            "descriptive_dsp_difference": 0 if pair == 20 else None,
            "status": ("OBSERVED_MATCHED_OOC_ROUTE_WITH_DISCLOSED_WARNINGS"
                       if pair == 20 else "UNKNOWN_OR_REJECTED_PHYSICAL_PAIR"),
        })
    (export / "summary.json").write_text(json.dumps({
        "schema": "fpgai-research/cst-R2j-original-routed-fanout-pair-export/v1",
        "source_revision": "c" * 40,
        "prespecified_cases": 72, "prespecified_pairs": 36,
        "original_vivado_invocation_receipts": 2,
    }))
    (export / "original_physical_audit.json").write_text(json.dumps({
        "prespecified_cases": 72, "cases": rows,
    }))
    (export / "matched_pairs.json").write_text(json.dumps(pairs))
    return root, export


def test_raw_drc_has_nine_dsp_instances_and_26_nets():
    violations = parse_drc_report(_raw_drc())
    assert len(violations) == 10
    assert sum(v["dsp_cell"] is not None for v in violations) == 9
    assert [v["no_routable_load_net_count"] for v in violations
            if v["rule"] == "RTSTAT-10"] == [26]


def test_frozen_original_pair_is_descriptive_not_e4(physical_fixture, tmp_path):
    physical, export = physical_fixture
    original = [(physical / f"case_{i:03d}" / "vivado" / "reports" /
                 "drc.rpt").read_bytes() for i in (40, 41)]
    report = forensic_report(physical_study=physical, export_study=export, pair=20)
    assert report["schema"] == SCHEMA
    assert report["same_rule_multiplicities"]
    assert report["same_reported_DSP_cell_identities"]
    assert report["same_RTSTAT_no_load_net_counts"]
    assert report["same_exact_RTSTAT_net_identities"] is None
    assert report["both_strict_clean_DRC_E4_qualified"] is False
    assert report["new_original_GCC_Vitis_Vivado_calls"] == 0
    assert all(row["count"] == 10 for row in report["cases"])
    assert original == [(physical / f"case_{i:03d}" / "vivado" / "reports" /
                         "drc.rpt").read_bytes() for i in (40, 41)]
    target = tmp_path / "prospective_drc.json"
    args = ["--physical-study", str(physical), "--export-study", str(export),
            "--pair", "20", "--output", str(target)]
    main(args)
    assert json.loads(target.read_text()) == report
    with pytest.raises(FileExistsError, match="create-only"):
        main(args)


def test_modified_original_report_rejected(physical_fixture):
    physical, export = physical_fixture
    path = physical / "case_041" / "vivado" / "reports" / "drc.rpt"
    path.write_text(path.read_text().replace("26 net(s)", "25 net(s)"))
    with pytest.raises(ValueError, match="differ from routed vendor receipt"):
        forensic_report(physical_study=physical, export_study=export, pair=20)


def test_summary_counterfeit_rejected(physical_fixture):
    physical, export = physical_fixture
    path = export / "original_physical_audit.json"
    audit = json.loads(path.read_text())
    audit["cases"][40]["drc_warning_summary"]["rules"]["DPIP-2"]["violations"] = 4
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Raw DRC violations contradict"):
        forensic_report(physical_study=physical, export_study=export, pair=20)


def test_original_vendor_receipt_disagreement_rejected(physical_fixture):
    physical, export = physical_fixture
    path = export / "original_physical_audit.json"
    audit = json.loads(path.read_text())
    audit["cases"][40]["original_routed_report_sha256"]["drc.rpt"] = "0" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="does not match Vivado invocation receipt"):
        forensic_report(physical_study=physical, export_study=export, pair=20)


def test_unobserved_pair_cannot_be_fabricated(physical_fixture):
    physical, export = physical_fixture
    with pytest.raises(ValueError, match="no matched original routed observations"):
        forensic_report(physical_study=physical, export_study=export, pair=19)


def test_duplicate_and_missing_dsp_witnesses_rejected():
    with pytest.raises(ValueError, match="missing its original cell identity"):
        parse_drc_report("DPIP-2#1 Warning\nPipelining DSP\nMissing path\n")
    with pytest.raises(ValueError, match="Duplicate numbered"):
        parse_drc_report("RTSTAT-10#1 Warning\n26 net(s) have no routable loads.\n"
                         "RTSTAT-10#1 Warning\n26 net(s) have no routable loads.\n")


def test_same_rule_counts_different_dsp_cells_are_not_declared_identical(physical_fixture):
    physical, export = physical_fixture
    drc = physical / "case_041" / "vivado" / "reports" / "drc.rpt"
    drc.write_text(drc.read_text().replace("grp_test/DPIP-2_U1", "grp_other/DPIP-2_U1"))
    digest = sha256(drc.read_bytes()).hexdigest()
    receipt_path = drc.parent.parent / "vivado_invocation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["fresh_report_sha256"]["drc.rpt"] = digest
    receipt_path.write_text(json.dumps(receipt))
    audit_path = export / "original_physical_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["cases"][41]["original_routed_report_sha256"]["drc.rpt"] = digest
    audit_path.write_text(json.dumps(audit))
    report = forensic_report(physical_study=physical, export_study=export, pair=20)
    assert report["same_rule_multiplicities"] is True
    assert report["same_reported_DSP_cell_identities"] is False
    assert report["both_strict_clean_DRC_E4_qualified"] is False
