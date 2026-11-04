"""Read-only hls4ml 1.3.0 configuration-level causality preflight.

The ONNX config generator expands non-reuse LayerName attributes including
Precision/Strategy; a comparison called 'RF1 versus RF2' is not automatically
a reuse-only comparison when configuration granularity changes.
This audit uses immutable existing E1 receipts only, with no vendor calls.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
from tempfile import TemporaryDirectory

SCHEMA = "fpgai-research/hls4ml-name-config-delta/v1"


def _read_e1(path):
    folder = Path(path)
    file = folder / "research_baseline.json"
    if (folder.is_symlink() or not folder.is_dir() or file.is_symlink()
            or not file.is_file() or file.stat().st_size > 8*1024*1024):
        raise ValueError("Missing safe private E1 baseline receipt")
    record = json.loads(file.read_text(encoding="utf-8"))
    if (not isinstance(record, dict)
            or record.get("schema") != "fpgai-research/hls4ml-baseline/v2"
            or record.get("evidence") != "E1-source-only"
            or not isinstance(record.get("hls_config"), dict)):
        raise ValueError("Not an attested E1 configuration receipt")
    return record, sha256(file.read_bytes()).hexdigest()


def compare_e1_configuration(global_default, name_rf1, name_rf2):
    trio = [_read_e1(folder) for folder in (global_default, name_rf1, name_rf2)]
    g, one, two = [item[0] for item in trio]
    must_match = ("model_sha256","part","clock_ns","precision","backend","io_type",
                  "hls4ml_version","reuse_factor")
    if any(any(obj.get(field) != g.get(field) for field in must_match)
           for obj in (one,two)):
        raise ValueError("E1 candidate, target or precision differs across configurations")
    if (g.get("configuration_granularity") != "model"
            or one.get("configuration_granularity") != "name"
            or two.get("configuration_granularity") != "name"):
        raise ValueError("Expected global default, name RF1, name RF2")
    layers1 = one["hls_config"].get("LayerName")
    layers2 = two["hls_config"].get("LayerName")
    if (not isinstance(layers1, dict) or not isinstance(layers2, dict)
            or set(layers1) != set(layers2)
            or not all(isinstance(x, dict) for x in (*layers1.values(),*layers2.values()))):
        raise ValueError("Name-level layer sets are not identical")
    if g["hls_config"].get("Model") != one["hls_config"].get("Model") or (
            g["hls_config"].get("Model") != two["hls_config"].get("Model")):
        raise ValueError("Base hls4ml Model-level configuration differs")
    differences = [
        (name, key)
        for name in sorted(layers1)
        for key in set(layers1[name]) | set(layers2[name])
        if layers1[name].get(key) != layers2[name].get(key)
    ]
    # A source receipt records which layer(s) the researcher explicitly
    # changed. Distinguish these from any other inferred LayerName deltas.
    # Earlier runs changed two reuse keys despite a one-layer request.
    requested_one = one.get("requested_layer_reuse", {})
    requested_two = two.get("requested_layer_reuse", {})
    if (not isinstance(requested_one, dict)
            or not isinstance(requested_two, dict)):
        raise ValueError("Malformed requested layer reuse provenance")
    selected_names = set(requested_one) | set(requested_two)
    expected_changes = {
        (name, "ReuseFactor") for name in selected_names
        if requested_one.get(name) != requested_two.get(name)
    }
    observed_changes = set(differences)
    unexpected_changes = observed_changes - expected_changes
    missing_changes = expected_changes - observed_changes
    only_explicit_reuse_changed = (
        len(selected_names) == 1 and len(expected_changes) == 1
        and not unexpected_changes and not missing_changes
        and all(key == "ReuseFactor" for _, key in differences)
    )
    # Compare the *actual requested* hls4ml configuration independently of
    # the post-conversion config. hls4ml may insert or mutate defaults while
    # building ModelGraph; that side effect is a numerical diagnostic, NOT
    # permission to describe the two generated circuits as reuse-only.
    before_one = one.get("hls_config_requested_before_conversion")
    before_two = two.get("hls_config_requested_before_conversion")
    requested_config_available = (
        isinstance(before_one, dict) and isinstance(before_two, dict)
        and isinstance(before_one.get("LayerName"), dict)
        and isinstance(before_two.get("LayerName"), dict)
        and isinstance(before_one.get("Model"), dict)
        and before_one["Model"] == before_two["Model"]
    )
    requested_changed = None
    requested_clean = False
    backend_extra = None
    if requested_config_available:
        names_before1 = before_one["LayerName"]
        names_before2 = before_two["LayerName"]
        requested_changed = {
            (name, key)
            for name in set(names_before1) | set(names_before2)
            for key in set(names_before1.get(name, {})) | set(names_before2.get(name, {}))
            if names_before1.get(name, {}).get(key)
               != names_before2.get(name, {}).get(key)
        }
        # Same layer namespace, one intended MatMul reuse delta; forbid
        # any requested precision/strategy change.
        requested_clean = (
            len(selected_names) == 1
            and len(expected_changes) == 1
            and set(names_before1) == set(names_before2)
            and requested_changed == expected_changes
        )
        backend_extra = len(observed_changes - requested_changed)
    # A rejected configuration needs a usable, private-name-free diagnosis.
    # The sorted index is stable for THIS source-bound inventory but is not
    # a model layer name. Values are disclosed only for integer RF settings.
    anonymous_deltas = []
    for index, name in enumerate(sorted(layers1)):
        for key in sorted(set(layers1[name]) | set(layers2[name])):
            left, right = layers1[name].get(key), layers2[name].get(key)
            if left == right:
                continue
            rf_values = (
                key == "ReuseFactor" and type(left) is int
                and type(right) is int
            )
            anonymous_deltas.append({
                "layer_index_in_sorted_e1_names": index,
                "key": key,
                "selected_explicitly": name in selected_names,
                "expected_explicit_change": (name, key) in expected_changes,
                "rf1_value": left if rf_values else None,
                "rf2_value": right if rf_values else None,
                "confidential_layer_name_recorded": False,
            })
    def count_key(key):
        return sum(int(key in item) for item in layers1.values())
    def audit_source(item):
        rows=item.get("generated_cpp_source_inventory")
        if not isinstance(rows,list):
            raise ValueError("Missing generated E1 source inventory")
        output={}
        for row in rows:
            if not isinstance(row,dict) or "relative_path" not in row or "sha256" not in row:
                raise ValueError("Unrecognized E1 source inventory")
            output[row["relative_path"]]=row["sha256"]
        return output
    gsource,one_source,two_source=map(audit_source,(g,one,two))
    def changed(a,b):
        return sum(a.get(k)!=b.get(k) for k in set(a)|set(b))
    return {
        "schema":SCHEMA,
        "comparison_model_source_identical":True,
        "same_model_level_configuration":True,
        "native_execution_or_e2_qualified":False,
        "global_e1_receipt_sha256":trio[0][1],
        "name_rf1_e1_receipt_sha256":trio[1][1],
        "name_rf2_e1_receipt_sha256":trio[2][1],
        "name_layer_count":len(layers1),
        "name_layers_with_precision_overrides":count_key("Precision"),
        "name_layers_with_strategy_overrides":count_key("Strategy"),
        "name_layers_with_reuse_factor":count_key("ReuseFactor"),
        "name_rf1_vs_rf2_changed_configuration_key_count":len(differences),
        "name_rf1_vs_rf2_changed_non_reuse_key_count":
            sum(key!="ReuseFactor" for _,key in differences),
        "name_rf1_vs_rf2_changed_reuse_key_count":
            sum(key=="ReuseFactor" for _,key in differences),
        "name_rf1_vs_rf2_selected_requested_layer_count":len(selected_names),
        "name_rf1_vs_rf2_expected_requested_reuse_changes":len(expected_changes),
        "name_rf1_vs_rf2_unexpected_reuse_or_other_delta_count":len(unexpected_changes),
        "name_rf1_vs_rf2_missing_requested_delta_count":len(missing_changes),
        "name_rf1_vs_rf2_only_explicit_reuse_changed":only_explicit_reuse_changed,
        "preconversion_requested_hls_config_available":requested_config_available,
        "preconversion_only_explicit_reuse_changed":requested_clean,
        "preconversion_changed_key_count":
            len(requested_changed) if requested_changed is not None else None,
        "postconversion_additional_changed_key_count":backend_extra,
        "postconversion_isolation_lost":
            bool(requested_clean and not only_explicit_reuse_changed),
        "hardware_reuse_only_comparison_qualified":
            bool(requested_clean and only_explicit_reuse_changed),
        "name_rf1_vs_rf2_anonymous_changed_settings":anonymous_deltas,
        "global_vs_name_rf1_generated_cpp_source_changed_count":changed(gsource,one_source),
        "name_rf1_vs_rf2_generated_cpp_source_changed_count":changed(one_source,two_source),
        "configuration_causality_or_first_bad_layer_proven":False,
        "native_hardware_superiority_eligible":False,
    }


def export_e1_configuration_delta(global_default,name_rf1,name_rf2,output,
                                  *, require_only_selected_reuse=False):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable config audit destination exists")
    result=compare_e1_configuration(global_default,name_rf1,name_rf2)
    if require_only_selected_reuse and not result["name_rf1_vs_rf2_only_explicit_reuse_changed"]:
        raise ValueError(
            "RF1/RF2 comparison changed settings beyond the explicitly "
            "requested ReuseFactor control; refusing a reuse-only claim")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".hls4ml-config-delta-",dir=dest.parent) as tmp:
        stage=Path(tmp)/"audit"
        stage.mkdir()
        (stage/"config_delta_private.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Config audit destination appeared")
        stage.rename(dest)
    return result
