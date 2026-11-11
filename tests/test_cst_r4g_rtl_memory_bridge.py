"""R4g read-only original RTL source-graph bridge regression tests.

Synthetic Verilog and invocation receipts are parser tests ONLY; no Vitis,
Vivado, native compilation, or real hardware evidence is asserted.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4g_rtl_memory_bridge as sut


FULL = """module fpgai_model();
  scratch_RAM u_store();
  helper u_math();
endmodule
module scratch_RAM();
  reg [31:0] scratch [0:255];
endmodule
module helper();
endmodule
"""
ROW = """module fpgai_model();
  scratch_RAM u_store();
  helper u_math();
endmodule
module scratch_RAM();
  reg [31:0] scratch [0:15];
endmodule
module helper();
endmodule
"""


def _observation(arm, format, *, e3):
    return {
        "arm": arm, "format": list(format),
        "E3_qualified": e3, "E2_qualified": e3,
        "original_source_sha256": "a" * 64 if arm == "full_H" else "b" * 64,
        "HLS_estimated_resources": {
            "bram_18k": 13, "dsp": 23 if arm == "full_H" else 19,
            "lut": 4990 if arm == "full_H" else 4755,
            "ff": 1798 if arm == "full_H" else 1647, "uram": 0,
        } if e3 else {key: None for key in ("bram_18k","dsp","lut","ff","uram")},
    }


def _native():
    return {
        "source_attested_arms": 6,
        "source_revision": "a" * 40,
        "original_R4a_source_revision": "b" * 40,
        "E3_qualified_pairs": 1,
        "observations": [
            _observation("full_H", (16, 6), e3=False),
            _observation("row_H", (16, 6), e3=False),
            _observation("full_H", (32, 6), e3=True),
            _observation("row_H", (32, 6), e3=True),
            _observation("full_H", (32, 8), e3=False),
            _observation("row_H", (32, 8), e3=False),
        ],
    }


def _make_original_native(pkg, observation, content):
    rtl = pkg / "fpgai_hls_project/solution1/syn/verilog"
    rtl.mkdir(parents=True)
    source = rtl / "fpgai_model.v"
    source.write_text(content)
    receipt = {
        "schema": "fpgai-research/vendor-invocation/v1",
        "tool": "vitis_hls",
        "source_sha256": observation["original_source_sha256"],
        "completed_successfully": True,
        "inputs_stable_during_execution": True,
        "rtl_output_sha256": {"fpgai_model.v": sha256(source.read_bytes()).hexdigest()},
        "rom_output_sha256": {},
    }
    (pkg / "vitis_invocation.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n")
    return source


def test_valid_original_source_rtl_is_only_a_static_candidate(tmp_path):
    full_obs = _observation("full_H", (32, 6), e3=True)
    row_obs = _observation("row_H", (32, 6), e3=True)
    p1, p2 = tmp_path / "full", tmp_path / "row"
    f1 = _make_original_native(p1, full_obs, FULL)
    f2 = _make_original_native(p2, row_obs, ROW)
    sha_before = (sha256(f1.read_bytes()).hexdigest(),
                  sha256(f2.read_bytes()).hexdigest())
    a = sut.inspect_arm(p1, observation=full_obs)
    b = sut.inspect_arm(p2, observation=row_obs)
    assert a["rtl_sha_attested"] and b["rtl_sha_attested"]
    assert a["rtl_file_count"] == b["rtl_file_count"] == 1
    assert a["rom_file_count"] == b["rom_file_count"] == 0
    assert a["source_module_parse_status"] == "STATIC_TYPE_GRAPH_ONLY"
    assert a["static"]["reachable_scratch_module_type_count"] == 1
    assert b["static"]["reachable_scratch_module_type_count"] == 1
    assert a["static"]["reachable_memory_module_type_instances"] == 1
    paired = sut.compare_pair(a, b, format=[32, 6])
    assert paired["HLS_estimated_BRAM18K_delta_row_minus_full"] == 0
    assert paired["both_static_type_graphs_known"] is True
    assert paired["candidate_RTL_structure_delta_row_minus_full"][
        "reachable_memory_module_type_instances"] == 0
    assert paired["physical_BRAM_delta"] is None
    assert a["physical_BRAM"] is None
    assert sha_before == (sha256(f1.read_bytes()).hexdigest(),
                          sha256(f2.read_bytes()).hexdigest())


def test_missing_natively_unattempted_formats_stay_unknown(tmp_path):
    no_native = _observation("full_H", (32, 8), e3=False)
    outcome = sut.inspect_arm(tmp_path / "not_run", observation=no_native)
    assert outcome["rtl_observation_status"] == "UNKNOWN_E3_NOT_QUALIFIED"
    assert outcome["rtl_file_count"] is None
    assert outcome["static"]["reachable_memory_module_type_instances"] is None
    row = sut.inspect_arm(
        tmp_path / "row_not_run",
        observation=_observation("row_H", (32, 8), e3=False))
    pair = sut.compare_pair(outcome, row, format=[32, 8])
    assert pair["HLS_estimated_BRAM18K_delta_row_minus_full"] is None
    assert pair["candidate_RTL_structure_delta_row_minus_full"][
        "reachable_memory_module_type_instances"] is None


def test_source_tamper_and_symlink_rejected(tmp_path):
    obs = _observation("full_H", (32, 6), e3=True)
    pkg = tmp_path / "valid"
    file = _make_original_native(pkg, obs, FULL)
    file.write_text(file.read_text() + "\n// AFTER_NATIVE\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        sut.inspect_arm(pkg, observation=obs)
    new = tmp_path / "linked"
    new.symlink_to(pkg, target_is_directory=True)
    with pytest.raises(ValueError, match="missing/symlinked"):
        sut.inspect_arm(new, observation=obs)


def test_no_rtl_is_explicit_unknown(tmp_path):
    obs = _observation("full_H", (32, 6), e3=True)
    pkg = tmp_path / "case"
    pkg.mkdir()
    (pkg / "vitis_invocation.json").write_text(json.dumps({
        "schema": "fpgai-research/vendor-invocation/v1",
        "tool": "vitis_hls", "source_sha256": "a" * 64,
        "completed_successfully": True,
        "inputs_stable_during_execution": True,
        "rtl_output_sha256": {}, "rom_output_sha256": {},
    }))
    result = sut.inspect_arm(pkg, observation=obs)
    assert result["rtl_sha_attested"] is False
    assert result["rtl_observation_status"] == "UNKNOWN_MISSING_GENERATED_RTL"
    assert result["rtl_file_count"] is None


def test_unsupported_static_verilog_is_unknown_not_zero(tmp_path):
    obs = _observation("full_H", (32, 6), e3=True)
    pkg = tmp_path / "case"
    _make_original_native(pkg, obs, "module other(); endmodule")
    result = sut.inspect_arm(pkg, observation=obs)
    assert result["rtl_sha_attested"] is True
    assert result["rtl_observation_status"] == "UNKNOWN_UNSUPPORTED_STATIC_VERILOG"
    assert result["static"]["reachable_memory_module_type_count"] is None


def test_entire_prespecified_population_and_create_only_report(
    tmp_path, monkeypatch,
):
    root = tmp_path / "original_new_r4d"
    original = tmp_path / "original_r4a"
    root.mkdir()
    original.mkdir()
    observations = _native()
    for obs in observations["observations"]:
        if obs["E3_qualified"]:
            bits, integer = obs["format"]
            pkg = (root / f"format_{bits:02d}_{integer:02d}"
                   / obs["arm"] / "vitis_source")
            _make_original_native(pkg, obs, FULL if obs["arm"] == "full_H"
                                   else ROW)
    monkeypatch.setattr(sut.r4e, "audit_native", lambda **_: observations)
    report = sut.inspect(study=root, original_study=original)
    assert report["prespecified_pairs"] == 3
    assert report["native_E3_qualified_pairs"] == 1
    assert report["original_RTL_SHA_attested_arms"] == 2
    assert len(report["arms"]) == 6
    assert report["static_type_graph_pairs_known"] == 1
    assert report["pairs"][0]["HLS_estimated_BRAM18K_delta_row_minus_full"] is None
    assert report["pairs"][1]["HLS_estimated_BRAM18K_delta_row_minus_full"] == 0
    assert report["pairs"][2]["HLS_estimated_BRAM18K_delta_row_minus_full"] is None
    assert report["physical_E4"] is None
    out = tmp_path / "new_report"
    sut.run(study=root, original_study=original, output=out)
    actual = (out / "report.json").read_bytes()
    digests = json.loads((out / "artifact_sha256.json").read_text())
    assert digests["report.json"] == sha256(actual).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(study=root, original_study=original, output=out)
    with pytest.raises(ValueError, match="inside original"):
        sut.run(study=root, original_study=original,
                output=root / "forbidden")
