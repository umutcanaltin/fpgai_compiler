"""Pre-run environment readiness is diagnostic, read-only, and fail-closed."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

def _load():
    script=Path(__file__).resolve().parents[1]/"experiments/compiler_ir/doctor.py"
    spec=importlib.util.spec_from_file_location("compiler_ir_doctor",script)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def test_local_machine_report_does_not_execute_native_tools(tmp_path, monkeypatch):
    mod=_load()
    calls=[]
    def probe(tool, args, timeout=15):
        calls.append((tool,args))
        return {"available":tool!="vitis_hls","resolved_path":tool,
                "version":"local-probe"}
    monkeypatch.setattr(mod,"_probe_command",probe)
    output=mod.inspect(Path(__file__).resolve().parents[1],
                       with_native=False)
    assert output["schema"].endswith("/v1")
    assert not output["native_binary_detected"]
    assert not output["native_execution_environment_ready_to_attempt"]
    assert output["trained_assets"]["mlp"]["requested"] is False
    assert all(name!="vitis_hls" for name, _args in calls)
    assert output["git_revision"]
    assert output["disk_free_gib"] >= 0
    assert not output["caveats"][0].startswith("Native synthesized")


def test_explicit_native_preflight_checks_binary_without_synthesis(monkeypatch):
    mod=_load()
    calls=[]
    def fake_probe(name,args,timeout=15):
        calls.append((name,args))
        return {"available": name != "vitis_hls",
                "resolved_path":name, "version":None}
    monkeypatch.setattr(mod,"_probe_command",fake_probe)
    info=mod.inspect(Path(__file__).resolve().parents[1],
                     with_native=True)
    assert ("vitis_hls",("-version",)) in calls
    assert not info["native_binary_detected"]
    assert not info["native_execution_environment_ready_to_attempt"]


def test_required_original_trained_assets_report_missing_without_fabrication(tmp_path):
    mod=_load()
    info=mod.inspect(Path(__file__).resolve().parents[1],
                     trained_mlp=str(tmp_path/"missing"))
    asset=info["trained_assets"]["mlp"]
    assert asset["requested"] is True
    assert asset["ready"] is False
    assert "Missing" in asset["reason"]
    assert not (tmp_path/"missing").exists()
