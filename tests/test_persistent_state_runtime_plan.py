from pathlib import Path

from fpgai.ir.graph import Graph
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state
from fpgai.runtime.package_builder import emit_runtime_package
from fpgai.runtime.runtime_plans import build_persistent_state_plan


def _graph() -> Graph:
    g = Graph("stateful")
    g.add_tensor("k_cache", (1, 3, 16, 64), "float32")
    g.add_tensor("v_cache", (1, 3, 16, 64), "float32")
    configure_kv_cache_state(g, key_cache="k_cache", value_cache="v_cache", capacity=16, sequence_axis=2, storage="uram")
    return g


def test_persistent_state_plan_is_generic_and_records_kv_semantics() -> None:
    plan = build_persistent_state_plan(_graph())
    assert plan["schema"] == "fpgai.persistent-state-plan/v1"
    assert plan["tensor_count"] == 2
    assert plan["backend_required"] is True
    assert plan["required_operations"] == ["reset", "import", "export", "read", "write"]
    by_name = {row["name"]: row for row in plan["tensors"]}
    assert by_name["k_cache"]["kind"] == "kv_key_cache"
    assert by_name["k_cache"]["storage"] == "uram"
    assert by_name["k_cache"]["lifetime"] == "runtime_session"
    assert by_name["k_cache"]["update_policy"] == "append"
    assert by_name["k_cache"]["capacity"] == 16


def test_runtime_package_preserves_persistent_state_plan(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    out.mkdir()
    state = build_persistent_state_plan(_graph())
    emit_runtime_package(out, pipeline_mode="inference", top_name="deeplearn", persistent_state_plan=state)
    import json
    manifest = json.loads((out / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["persistent_state"]["tensor_count"] == 2
    # A later packaging refresh must not erase compiler-resolved state metadata.
    emit_runtime_package(out)
    refreshed = json.loads((out / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert refreshed["persistent_state"] == manifest["persistent_state"]
