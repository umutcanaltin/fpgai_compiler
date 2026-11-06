"""R4n new R4l RTL-memory audit: synthetic Verilog, NEVER run vendor tools.

Exercises exact source hash attestation and bounded 3-site instance
interpretation. All generated test Verilog is synthetic and provides
NO new hardware resource or physical FPGA evidence.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4n_schedule_matched_rtl_memory as sut
from fpgai_research.experiments import cst_r4g_rtl_memory_bridge as r4g

SCRATCH = "fpgai_model_scratch_RAM_AUTO_1R1W"
ACC = "fpgai_model_acc_RAM_AUTO_1R1W"
SOURCE_HASHES = {
    "full_storage": "03b0cc275ba27294d99adb14d597717d2a9035853b572b4dff6d7e6dcc4df98f",
    "row_storage": "aa0a7a2f684f3c0e6e64f6b4e3084c8f0dc93ae152318bd5339d9646b4894e9e",
}


def _source(depth, *, override=None):
    literal = depth if override is None else override
    return {
        "fpgai_model.v": (
            "module fpgai_model(input clk);\n"
            + "\n".join(
                f"{SCRATCH} #(.DataWidth(32), .AddressWidth(8), "
                f".AddressRange({literal})) {name} (.clk(clk));"
                for name in ("scratch_U", "scratch_1_U", "scratch_2_U")
            ) + "\n"
            f"{ACC} #(.DataWidth(32), .AddressWidth(8), "
            ".AddressRange(256)) acc_U(.clk(clk));\n"
            "endmodule\n"
        ),
        SCRATCH + ".v": (
            f"module {SCRATCH}(input clk);\n"
            "parameter DataWidth = 32;\n"
            "parameter AddressWidth = 8;\n"
            f"parameter AddressRange = {depth};\n"
            "reg [DataWidth-1:0] ram[0:AddressRange-1];\n"
            "endmodule\n"
        ),
        ACC + ".v": (
            f"module {ACC}(input clk);\n"
            "parameter DataWidth = 32;\n"
            "parameter AddressWidth = 8;\n"
            "parameter AddressRange = 256;\n"
            "reg [DataWidth-1:0] ram[0:AddressRange-1];\n"
            "endmodule\n"
        ),
    }


@pytest.fixture
def fake_originals(tmp_path, monkeypatch):
    new, old_d, old_a = (tmp_path / "new_r4k",
                         tmp_path / "old_r4d",
                         tmp_path / "old_r4a")
    for folder in (new, old_d, old_a):
        folder.mkdir()
    receipts = {}
    observations = []
    for arm, depth, bram in (
        ("full_storage", 214, 13),
        ("row_storage", 134, 13),
    ):
        pkg = new / arm / "vitis_source"
        folder = pkg / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
        folder.mkdir(parents=True)
        sources = _source(depth)
        hashes = {}
        for name, content in sources.items():
            file = folder / name
            file.write_text(content)
            hashes[name] = sha256(file.read_bytes()).hexdigest()
        receipts[arm] = {
            "rtl_file_sha256": hashes,
            "rom_file_sha256": {"weight.dat": "f" * 64},
            "rtl_file_count": len(hashes),
            "rom_file_count": 1,
            "static": {key: 4 for key in r4g.STATIC_KEYS},
            "rtl_observation_status": "ATTESTED_STATIC_TYPE_GRAPH_ONLY",
            "provisionally_reachable_memory_module_types": [
                SCRATCH, ACC],
        }
        observations.append({
            "arm": arm, "E2_qualified": True, "E3_qualified": True,
            "source_sha256": SOURCE_HASHES[arm],
            "HLS_estimated_resources": {"bram_18k": bram},
        })
    native = {
        "E1_matched_pair_accepted": True,
        "paired_HLS_source_arms": 2,
        "observations": observations,
        "original_R4a_source_revision": "a" * 40,
        "original_R4d_source_revision": "d" * 40,
        "source_revision": "k" * 40,
        "both_E2_qualified": True,
        "both_E3_qualified": True,
    }
    monkeypatch.setattr(sut.r4l, "audit_native",
                        lambda **kw: native)
    def original_r4g(pkg, *, observation):
        arm = observation["arm"]
        assert observation["original_source_sha256"] == SOURCE_HASHES[arm]
        assert pkg == new / arm / "vitis_source"
        return receipts[arm]
    monkeypatch.setattr(sut.r4g, "inspect_arm", original_r4g)
    return (new, old_d, old_a), native, receipts


def _inspect(inputs):
    (new, old_d, old_a), _, _ = inputs
    return sut.inspect(r4k_study=new, original_r4d=old_d,
                       original_r4a=old_a)


def test_both_new_original_rtl_source_memory_sites_and_bram_tie(fake_originals):
    data = _inspect(fake_originals)
    assert data["original_E2_E3_qualified_pair"] is True
    assert data["R4l_original_RTL_source_arms_attested"] == 2
    assert data["native_vendor_calls_during_read_only_analysis"] == 0
    assert data["original_evidence_modified"] is False
    assert data["physical_E4"] is None
    pair = data["pair"]
    assert pair["status"] == (
        "SOURCE_ATTESTED_STATIC_RTL_MEMORY_PAIR_NOT_ELABORATED")
    assert pair["both_RTL_SHA_attested"] is True
    assert pair["HLS_estimated_BRAM18K_delta_row_minus_full"] == 0
    assert pair["ROM_content_sha_multiset_identical"] is True
    assert pair["scratch_source_depth_delta_row_minus_full"] == -80
    source = pair["scratch_source_parameter_sites"]
    assert source["scratch_source_site_count_full"] == 3
    assert source["scratch_source_site_count_row"] == 3
    assert len(source["scratch_source_site_pairs"]) == 3
    assert all(site["source_depth_delta_row_minus_full"] == -80
               for site in source["scratch_source_site_pairs"])
    assert source["elaborated_scratch_instance_depth_delta"] is None
    assert pair["mapped_BRAM18K_delta"] is None
    assert pair["placed_routed_clock_delta"] is None


def test_module_defaults_and_source_literal_instantiation_distinguished(
    fake_originals,
):
    rows = _inspect(fake_originals)["arms"]
    for arm, depth in zip(rows, (214, 134)):
        assert arm["status"] == (
            "ATTESTED_STATIC_SOURCE_AND_MEMORY_SITES_NOT_ELABORATED")
        profiles = arm["memory_parameters"]["source_parameter_defaults_known"]
        assert profiles[SCRATCH]["default_AddressRange_entries"] == depth
        assert profiles[SCRATCH]["default_DataWidth_bits"] == 32
        assert profiles[ACC]["default_AddressRange_entries"] == 256
        sites = arm["source_sites"]["scratch_source_sites"]
        assert len(sites) == 3
        assert all(site["source_site_literal_parameter_tuple_if_supported"] ==
                   {"DataWidth": 32, "AddressWidth": 8,
                    "AddressRange": depth} for site in sites)
        assert all(site["effective_elaborated_instance_parameters"] is None
                   for site in sites)


def test_source_sha_tampering_refused_before_comparison(fake_originals):
    (new, _, _), _, _ = fake_originals
    file = (new / "row_storage" / "vitis_source"
            / "fpgai_hls_project" / "solution1" / "syn"
            / "verilog" / "fpgai_model.v")
    file.write_bytes(file.read_bytes() + b"// edited after original invocation\n")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        _inspect(fake_originals)


def test_original_unknown_e3_never_called_into_static_rtl(
    fake_originals, monkeypatch,
):
    (new, _, _), native, _ = fake_originals
    native["observations"][0]["E3_qualified"] = False
    def guarded(pkg, *, observation):
        assert pkg == new / "row_storage" / "vitis_source"
        return fake_originals[2]["row_storage"]
    monkeypatch.setattr(sut.r4g, "inspect_arm", guarded)
    result = _inspect(fake_originals)
    assert result["arms"][0]["status"] == "UNKNOWN_E2_OR_E3_NOT_QUALIFIED"
    assert result["pair"]["status"].startswith("UNKNOWN")
    assert result["pair"]["scratch_source_depth_delta_row_minus_full"] is None
    assert result["pair"]["HLS_estimated_BRAM18K_delta_row_minus_full"] is None


def test_supported_literal_source_override_can_cancel_default_difference(
    fake_originals,
):
    (new, _, _), _, records = fake_originals
    # This is a synthetic case. Original 214/134 module defaults remain
    # different, but the new row source explicitly overrides its instances.
    file = (new / "row_storage" / "vitis_source"
            / "fpgai_hls_project" / "solution1" / "syn"
            / "verilog" / "fpgai_model.v")
    file.write_text(_source(134, override=214)["fpgai_model.v"])
    records["row_storage"]["rtl_file_sha256"][file.name] = (
        sha256(file.read_bytes()).hexdigest())
    result = _inspect(fake_originals)
    assert result["pair"]["scratch_source_depth_delta_row_minus_full"] == 0
    assert result["arms"][1]["memory_parameters"][
        "source_parameter_defaults_known"][SCRATCH][
            "default_AddressRange_entries"] == 134


def test_symbolic_source_override_fails_closed_but_rtl_stays_attested(
    fake_originals,
):
    (new, _, _), _, records = fake_originals
    file = (new / "row_storage" / "vitis_source"
            / "fpgai_hls_project" / "solution1" / "syn"
            / "verilog" / "fpgai_model.v")
    file.write_text(_source(134, override="SOME_DEPTH")["fpgai_model.v"])
    records["row_storage"]["rtl_file_sha256"][file.name] = (
        sha256(file.read_bytes()).hexdigest())
    result = _inspect(fake_originals)
    assert result["arms"][1]["status"] == (
        "ATTESTED_STATIC_SOURCE_WITH_UNKNOWN_MEMORY_SITES")
    assert result["pair"]["both_RTL_SHA_attested"] is True
    assert result["pair"]["scratch_source_depth_delta_row_minus_full"] is None
    assert result["pair"]["HLS_estimated_BRAM18K_delta_row_minus_full"] == 0
    assert result["pair"]["mapped_BRAM18K_delta"] is None


def test_missing_original_memory_source_file_refuses_false_default(
    fake_originals,
):
    (_, _, _), _, records = fake_originals
    records["row_storage"]["provisionally_reachable_memory_module_types"] = [
        SCRATCH, ACC, "fpgai_model_unknown_RAM_AUTO_1R1W"]
    rows = _inspect(fake_originals)["arms"]
    assert rows[1]["memory_parameters"]["unknown_parameter_modules"] == [
        "fpgai_model_unknown_RAM_AUTO_1R1W"]
    assert rows[1]["status"] == (
        "ATTESTED_STATIC_SOURCE_AND_MEMORY_SITES_NOT_ELABORATED")
    # A missing named memory module is NOT promoted to real capacity.
    assert rows[1]["memory_parameters"]["source_parameter_defaults_known"].get(
        "fpgai_model_unknown_RAM_AUTO_1R1W") is None


def test_attested_source_root_symlink_rejected(fake_originals, tmp_path):
    (new, _, _), _, _ = fake_originals
    root = (new / "full_storage" / "vitis_source"
            / "fpgai_hls_project" / "solution1" / "syn" / "verilog")
    extra = tmp_path / "external.v"
    extra.write_text("module external();endmodule\n")
    path = root / "link.v"
    path.symlink_to(extra)
    # A file added after attestation is not included in the signed SHA
    # set. Real R4g prevents this via inventory_vitis_rtl; the R4n
    # synthetic shim exercises only _signed_sources' explicit hash list.
    path.unlink()
    assert _inspect(fake_originals)["pair"]["both_RTL_SHA_attested"] is True


def test_r4n_output_is_create_only_and_never_updates_originals(
    fake_originals, tmp_path,
):
    (new, old_d, old_a), _, _ = fake_originals
    before = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for root in (new, old_d, old_a)
        for path in root.rglob("*") if path.is_file()
    }
    output = tmp_path / "r4n_new_readonly_report"
    report = sut.run(r4k_study=new, original_r4d=old_d,
                     original_r4a=old_a, output=output)
    assert report["native_vendor_calls_during_read_only_analysis"] == 0
    manifest = json.loads((output / "artifact_sha256.json").read_text())
    assert manifest["report.json"] == sha256(
        (output / "report.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(r4k_study=new, original_r4d=old_d,
                original_r4a=old_a, output=output)
    with pytest.raises(ValueError, match="outside original"):
        sut.run(r4k_study=new, original_r4d=old_d,
                original_r4a=old_a, output=new / "illegal_result")
    after = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for root in (new, old_d, old_a)
        for path in root.rglob("*") if path.is_file()
    }
    assert before == after


def test_wrong_arm_population_rejected(fake_originals):
    _, native, _ = fake_originals
    native["observations"][1]["arm"] = "full_storage"
    with pytest.raises(ValueError, match="population"):
        _inspect(fake_originals)
