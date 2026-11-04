"""One entry point from original trained MNIST networks to native measurements.

Source/host checks run locally. AMD tools run ONLY with --execute.
No model training, synthetic substitutes, benchmark fabrication, or mutation
of prior native evidence. Resume verifies the original input fingerprints.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.experiments.private_seven_axis_suite import (
    assemble_seven_axis_suite, SCHEMA as BUNDLE_SCHEMA,
)
from fpgai_research.experiments.all_operator_suite import (
    assemble_all_operator_suite, SCHEMA as OPERATORS_SCHEMA,
)
from fpgai_research.experiments.full_suite import prepare_suite, audit_suite
from fpgai_research.experiments.comprehensive_native_campaign import (
    execute_comprehensive_suite,
)
from fpgai_research.experiments.native_seven_axis_effects import (
    export_native_seven_axes,
)

DEFAULT_ASSETS = {
    "mlp_onnx": "build/trained_mlp_20261024/model.onnx",
    "mlp_input": "build/trained_mlp_20261024/inputs.npy",
    "cnn_onnx": "build/trained_cnn_20261024/model.onnx",
    "cnn_input": "build/trained_cnn_20261024/inputs.npy",
}
MAX_ASSET_BYTES = 128 * 1024 * 1024
SCHEMA = "fpgai-research/original-networks-workstation-experiment-run/v1"


def _original_hashes(paths):
    expected = set(DEFAULT_ASSETS)
    if not isinstance(paths, dict) or set(paths) != expected:
        raise ValueError("Provide both original trained ONNX networks and their original inputs")
    digests = {}
    for name, value in paths.items():
        path = Path(value)
        ext = ".onnx" if name.endswith("onnx") else ".npy"
        if (path.is_symlink() or not path.is_file() or path.suffix.lower() != ext
                or not 0 < path.stat().st_size <= MAX_ASSET_BYTES):
            raise FileNotFoundError("Missing/linked/oversized original model asset: " + name)
        digests[name] = sha256(path.read_bytes()).hexdigest()
    return digests


def _verified_bundle(bundle, original_hashes):
    path = bundle / "bundle_receipt.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original-network receipt missing")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (obj.get("schema") != BUNDLE_SCHEMA
            or obj.get("source_asset_sha256") != original_hashes
            or obj.get("models") != 2 or obj.get("weights_retrained") is not False
            or obj.get("native_e3_e4_executed") is not False):
        raise ValueError("Original network source receipt does not match these trained bytes")
    for key, filename in (
        ("mlp_onnx", "mlp.onnx"), ("mlp_input", "mlp.npy"),
        ("cnn_onnx", "cnn.onnx"), ("cnn_input", "cnn.npy"),
    ):
        item = bundle / "inputs" / filename
        if (item.is_symlink() or not item.is_file()
                or sha256(item.read_bytes()).hexdigest() != original_hashes[key]):
            raise ValueError("Original trained model/input bundle has been modified")
    return obj


def _verified_operators(bundle, operator_suite, original_hashes):
    record = operator_suite / "source_audit.json"
    if record.is_symlink() or not record.is_file():
        raise ValueError("All-operator source audit missing")
    obj = json.loads(record.read_text(encoding="utf-8"))
    if (obj.get("schema") != OPERATORS_SCHEMA
            or obj.get("original_source_asset_sha256") != original_hashes
            or obj.get("all_eligible_compute_operators_declared") is not True
            or obj.get("native_vendor_tools_executed") is not False
            or obj.get("source_bundle_receipt_sha256") !=
            sha256((bundle / "bundle_receipt.json").read_bytes()).hexdigest()):
        raise ValueError("All-operator source inventory does not match original models")
    for key, filename in (
        ("mlp_onnx", "mlp.onnx"), ("mlp_input", "mlp.npy"),
        ("cnn_onnx", "cnn.onnx"), ("cnn_input", "cnn.npy"),
    ):
        item = operator_suite / "inputs" / filename
        if (item.is_symlink() or not item.is_file()
                or sha256(item.read_bytes()).hexdigest() != original_hashes[key]):
            raise ValueError("All-operator network/input bytes do not match originals")
    return obj


def run_all_original_network_experiments(*, originals, workspace,
                                         execute=False, resume=False,
                                         start=0, max_cases=0,
                                         effects_output=None,
                                         vitis_executable="vitis_hls",
                                         vivado_executable="vivado"):
    if type(execute) is not bool or type(resume) is not bool:
        raise ValueError("Vendor execution and resume switches must be boolean")
    if type(start) is not int or type(max_cases) is not int or start < 0 or max_cases < 0:
        raise ValueError("Architecture window must be a nonnegative integer")
    hashes = _original_hashes(originals)
    root = Path(workspace)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("Workspace must be a real directory")
    if resume:
        if not root.is_dir():
            raise FileNotFoundError("Cannot resume missing experiment workspace")
    else:
        # Never accidentally reuse and silently requalify previous experiments.
        root.mkdir(parents=True, exist_ok=False)
    bundle = root / "original_trained_sources"
    operators = root / "all_compute_operators"
    prepared = root / "prepared_native_suite"
    journal = root / "native_execution_journal.json"
    if not bundle.exists():
        assemble_seven_axis_suite(
            originals["mlp_onnx"], originals["mlp_input"],
            originals["cnn_onnx"], originals["cnn_input"], bundle,
        )
    _verified_bundle(bundle, hashes)
    if not operators.exists():
        assemble_all_operator_suite(bundle, operators)
    operator_source = _verified_operators(bundle, operators, hashes)
    if not prepared.exists():
        prepare_suite(
            suite_path=operators / "suite.json",
            directory=prepared,
            host_samples=1,
            require_all_legal=False,
            crosscheck_onnxruntime=True,
            trace_intermediates=True,
        )
    audit = audit_suite(prepared, require_host=True, require_all_axes=False)
    if (audit.get("blocked_models") != 0 or
            not audit.get("all_prepared_models_have_legal_executable_subsets")):
        raise ValueError("One or more trained networks lacks audited executable cases")
    print("[FPGAI] Original trained assets and operator-local source "
          "audits verified; starting native case inventory", flush=True)
    native = execute_comprehensive_suite(
        prepared, execute=execute, resume=(execute and resume and journal.exists()),
        journal=journal if execute else None,
        start=start, max_cases=max_cases,
        vitis_executable=vitis_executable,
        vivado_executable=vivado_executable,
    )
    effects_written = None
    if effects_output is not None:
        # A post-run snapshot may record incomplete comparisons as null.
        # Never overwrite it on a subsequent resume.
        if not execute:
            raise ValueError("Effects export requires explicit native --execute")
        export_native_seven_axes(prepared, effects_output)
        effects_written = str(effects_output)
    return {
        "schema": SCHEMA,
        "original_asset_sha256": hashes,
        "operator_inventory": operator_source["operator_counts"],
        "declared_operator_studies": operator_source["total_operator_experiments"],
        "legal_case_count": native["total_legal_architecture_cases"],
        "excluded_infeasible_architecture_count":
            len(native["excluded_infeasible_architectures"]),
        "selected_case_count": native["selected_legal_cases"],
        "qualified_routed_cases_in_selection": native["native_routed_qualified_selected"],
        "all_selected_cases_routed_qualified": native["every_selected_case_routed_qualified"],
        "tools_executed_by_this_driver": native["vendor_tools_run_by_this_driver"],
        "workspace": str(root),
        "native_journal": str(journal) if execute else None,
        "native_effects_output": effects_written,
        "scientific_note": (
            "Full source coverage, complete routed coverage, FPGA board runtime, "
            "full-test fixed-point classification and matched physical hls4ml "
            "superiority must be verified independently; no missing results "
            "are imputed."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULT_ASSETS.items():
        p.add_argument("--" + key.replace("_", "-"), default=default)
    p.add_argument("--workspace", required=True)
    p.add_argument("--execute", action="store_true",
                   help="Explicitly invoke installed Vitis/Vivado in the foreground")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=0,
                   help="0 means every READY case")
    p.add_argument("--effects-output",
                   help="Fresh immutable native-effects output directory")
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--vivado-executable", default="vivado")
    args = p.parse_args(argv)
    paths = {key: getattr(args, key) for key in DEFAULT_ASSETS}
    report = run_all_original_network_experiments(
        originals=paths, workspace=args.workspace,
        execute=args.execute, resume=args.resume,
        start=args.start, max_cases=args.max_cases,
        effects_output=args.effects_output,
        vitis_executable=args.vitis_executable,
        vivado_executable=args.vivado_executable,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return int(args.execute and not report["all_selected_cases_routed_qualified"])


if __name__ == "__main__":
    raise SystemExit(main())
