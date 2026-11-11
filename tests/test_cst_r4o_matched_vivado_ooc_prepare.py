"""R4o safe two-arm physical input staging tests (NO Vivado execution)."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from fpgai_research.experiments import cst_r4o_matched_vivado_ooc_prepare as sut


@pytest.fixture
def original(tmp_path, monkeypatch):
    root, r4d, r4a = (tmp_path / "original_r4k",
                       tmp_path / "original_r4d",
                       tmp_path / "original_r4a")
    for folder in (root, r4d, r4a):
        folder.mkdir()
    observations = []
    for arm in sut.ARMS:
        pkg = root / arm / "vitis_source"
        inputs = (pkg / "fpgai_hls_project" / "solution1" / "syn"
                  / "verilog")
        inputs.mkdir(parents=True)
        files = {
            "fpgai_model.v": (
                "module fpgai_model(input ap_clk, output reg done);\n"
                "always @(posedge ap_clk) done <= 1;\n"
                "endmodule\n"),
            "weights.dat": "0a0a\n",
        }
        for name, body in files.items():
            (inputs / name).write_text(body)
        rtl = {"fpgai_model.v": sha256(
            (inputs / "fpgai_model.v").read_bytes()).hexdigest()}
        rom = {"weights.dat": sha256(
            (inputs / "weights.dat").read_bytes()).hexdigest()}
        observations.append({
            "arm": arm, "E2_qualified": True, "E3_qualified": True,
            "HLS_estimated_BRAM18K": 13,
            "source_cpp_sha256": "a" * 64 if arm == "full_storage"
                                  else "b" * 64,
            "rtl_source_receipt": {
                "status": "ATTESTED_STATIC_TYPE_GRAPH_ONLY",
                "rtl_file_sha256": rtl, "rom_file_sha256": rom},
        })
    raw = {
        "schema": sut.r4n.SCHEMA,
        "original_E2_E3_qualified_pair": True,
        "R4l_original_RTL_source_arms_attested": 2,
        "R4l_original_arms": 2,
        "arms": observations,
        "pair": {"both_RTL_SHA_attested": True,
                 "mapped_BRAM18K_delta": None},
        "R4k_original_source_revision": "k" * 40,
        "original_R4d_source_revision": "d" * 40,
        "original_R4a_source_revision": "a" * 40,
    }
    monkeypatch.setattr(sut.r4n, "inspect", lambda **kw: raw)
    return root, r4d, r4a, raw


def _args(original):
    root, r4d, r4a, _ = original
    return {"r4k_study": root, "original_r4d": r4d,
            "original_r4a": r4a}


def test_read_only_source_preflight_proves_both_original_hashes(original):
    result = sut.inspect(**_args(original))
    assert result["status"] == (
        "ORIGINAL_R4L_RTL_ROM_SOURCE_PINNED_READY_FOR_OOC_STAGING")
    assert result["both_R4l_E2_E3_qualified"] is True
    assert result["new_vivado_calls"] == 0
    assert result["physical_E4"] is None
    assert result["OOC_board_qualified"] is False
    assert list(result["arms"]) == list(sut.ARMS)
    assert all(r["original_rtl_file_count"] == 1
               and r["original_rom_file_count"] == 1
               for r in result["arms"].values())


def test_create_only_two_arm_rom_local_tcl_and_hash_inventory(
    original, tmp_path,
):
    args = _args(original)
    before = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for root in args.values() for p in root.rglob("*") if p.is_file()
    }
    output = tmp_path / "new_ooc"
    result = sut.prepare(**args, output=output)
    assert result["status"] == "SOURCE_ONLY_VIVADO_OOC_PREPARED_NOT_EXECUTED"
    assert result["new_vivado_calls"] == 0
    assert result["physical_E4"] is None
    for arm in sut.ARMS:
        case = output / arm
        source = (args["r4k_study"] / arm / "vitis_source" /
                  "fpgai_hls_project" / "solution1" / "syn" / "verilog")
        assert (case / "rtl" / "weights.dat").read_bytes() == (
            source / "weights.dat").read_bytes()
        tcl = (case / "run_vivado.tcl").read_text()
        assert "cd [file join $r4o_case rtl]" in tcl
        assert "read_verilog {fpgai_model.v}" in tcl
        assert "report_route_status" in tcl
        assert "report_timing_summary" in tcl
        assert "report_utilization" in tcl
        assert "report_drc" in tcl
        assert "synth_design -top fpgai_model -part xck26-sfvc784-2LV-c" in tcl
        assert "create_clock -name ap_clk -period 12.00" in tcl
        assert "set_clock_uncertainty 1.50" in tcl
        assert not (case / "reports").exists()
        assert not (case / "vivado_invocation.json").exists()
    manifest = json.loads((output / "artifact_sha256.json").read_text())
    assert manifest["summary.json"] == sha256(
        (output / "summary.json").read_bytes()).hexdigest()
    for path, digest in manifest.items():
        assert sha256((output / path).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="create-only"):
        sut.prepare(**args, output=output)
    after = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for root in args.values() for p in root.rglob("*") if p.is_file()
    }
    assert before == after


def test_original_rtl_or_rom_mutation_refused(original):
    args = _args(original)
    root = args["r4k_study"] / "row_storage" / "vitis_source"
    folder = root / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
    for file in ("fpgai_model.v", "weights.dat"):
        path = folder / file
        before = path.read_bytes()
        path.write_bytes(before + b"tampered\n")
        with pytest.raises(ValueError, match="missing or drifted"):
            sut.inspect(**args)
        path.write_bytes(before)


def test_original_rtl_symlink_rejected(original, tmp_path):
    args = _args(original)
    root = args["r4k_study"] / "row_storage" / "vitis_source"
    file = (root / "fpgai_hls_project" / "solution1" / "syn" /
            "verilog" / "fpgai_model.v")
    external = tmp_path / "external.v"
    external.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(external)
    with pytest.raises(ValueError):
        sut.inspect(**args)


def test_unqualified_native_or_missing_pair_refuses_staging(
    original, tmp_path,
):
    args = _args(original)
    raw = original[3]
    raw["original_E2_E3_qualified_pair"] = False
    with pytest.raises(ValueError, match="incomplete"):
        sut.prepare(**args, output=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
    raw["original_E2_E3_qualified_pair"] = True
    raw["R4l_original_RTL_source_arms_attested"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        sut.inspect(**args)


def test_fails_closed_when_target_inside_original_evidence(original):
    args = _args(original)
    with pytest.raises(ValueError, match="outside original study"):
        sut.prepare(**args, output=args["r4k_study"] / "do_not_create")
    assert not (args["r4k_study"] / "do_not_create").exists()


def test_invalid_tcl_file_names_rejected():
    with pytest.raises(ValueError, match="invalid Verilog"):
        sut._tcl(rtl_names=["../bad.v"])
    with pytest.raises(ValueError, match="invalid Verilog"):
        sut._tcl(rtl_names=[])
