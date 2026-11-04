"""Independent replay audit of a recorded paired hls4ml Vitis CSim diagnosis.

Rechecks original staged source/input fingerprints, native-run logfile SHA-256,
saved numerical observations and all 160 predictions without re-running Vitis.
A diagnostic with extra backend-added RF settings is NOT a fair RF-only sweep
and zero classification disagreements is NOT probability-equivalent E2.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_selected_native_pair import preflight
from fpgai_research.experiments.hls4ml_csim_native_observation import (
    discover_selected_csim_results,
)
from fpgai_research.experiments.hls4ml_csim_numeric import (
    collect_csim_numeric, numeric_matrix,
)
from fpgai_research.experiments.hls4ml_matched_numeric_forensics import (
    compare_matrices,
)

SCHEMA = "fpgai-research/hls4ml-native-pair-replay-audit/v1"
RUN_SCHEMA = "fpgai-research/hls4ml-selected-paired-native-csim/v1"
OBS_SCHEMA = "fpgai-research/hls4ml-selected-csim-numeric-observation/v1"


def _read_json(path: Path, limit=4 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Missing or unsafe private native-run receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Native-run receipt must be a JSON object")
    return value


def audit_native_pair(stage_root, global_default, native_run_dir):
    root = Path(stage_root)
    native = Path(native_run_dir)
    if native.is_symlink() or not native.is_dir():
        raise ValueError("Missing safe native observation directory")
    config = preflight(root, global_default, diagnostic_backend_effects=True)
    receipt_path = native / "paired_native_csim.json"
    recorded = _read_json(receipt_path)
    if (recorded.get("schema") != RUN_SCHEMA
            or recorded.get("native_executed") is not True
            or recorded.get("both_native_csim_completed") is not True
            or recorded.get("csim_only") is not True
            or recorded.get("native_e3_e4_executed") is not False
            or recorded.get("hardware_qor_comparison_eligible") is not False
            or recorded.get("qualified_e2") is not False
            or recorded.get("pair_source_config_reuse_only_qualified")
                is not config["hardware_reuse_only_comparison_eligible"]
            or recorded.get("diagnostic_only_not_fair_reuse_comparison")
                is not config["diagnostic_backend_effects_only"]
            or recorded.get("config_isolated") is not config["config_isolated"]):
        raise ValueError("Native run evidence scope or source configuration has changed")
    stages = recorded.get("stages")
    if (not isinstance(stages, list) or len(stages) != 2
            or any(not isinstance(x, str) for x in stages)):
        raise ValueError("Native run stage ancestry missing")
    for index, rf in enumerate((1, 2)):
        if Path(stages[index]).resolve(strict=True) != (
                root / f"csim_rf{rf}").resolve(strict=True):
            raise ValueError("Native receipt points at a different CSim workspace")
    attempts = recorded.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise ValueError("Native CSim did not record both RF attempts")
    numeric = []
    results = []
    verified_log_hashes = []
    for attempt, rf in zip(attempts, (1, 2)):
        log = native / f"rf{rf}_vitis_csim.log"
        if (not isinstance(attempt, dict) or attempt.get("rf") != rf
                or attempt.get("native_vitis_csim_executed") is not True
                or attempt.get("returncode") != 0
                or attempt.get("numerical_observation_16_rows") is not True
                or attempt.get("native_synthesis_or_vivado_executed") is not False
                or log.is_symlink() or not log.is_file()
                or log.stat().st_size > 16 * 1024 * 1024):
            raise ValueError("Native CSim run attempt or private log invalid")
        raw = log.read_bytes()
        if (sha256(raw).hexdigest() != attempt.get("log_sha256")
                or b"CSim done with 0 errors" not in raw):
            raise ValueError("Native CSim tool log fingerprint mismatch")
        verified_log_hashes.append(attempt["log_sha256"])

        stage = root / f"csim_rf{rf}"
        result_path = discover_selected_csim_results(stage)
        observed = collect_csim_numeric(stage, result_path)
        saved = _read_json(native / f"rf{rf}_numeric" / "observation.json")
        if saved != observed or saved.get("schema") != OBS_SCHEMA:
            raise ValueError("Stored numerical audit differs from original CSim result")
        measures = observed["measurements"]
        if (measures["selected_samples"] != 16
                or attempt.get("exploratory_numeric_policy_passed")
                    is not measures["selected_numeric_policy_passed"]
                or attempt.get("selected_top1_disagreements")
                    != measures["selected_top1_disagreements"]
                or attempt.get("max_absolute_probability_difference")
                    != measures["selected_probability_max_absolute_difference"]):
            raise ValueError("Native runner summary differs from source-bound numeric audit")
        numeric.append(observed)
        results.append(numeric_matrix(result_path.read_text(encoding="utf-8")))

    reference = numeric_matrix(
        (root / "csim_rf1" / "project" / "tb_data" /
         "tb_output_predictions.dat").read_text(encoding="utf-8"))
    comparison = compare_matrices(reference, results[0], results[1])
    if (comparison["default"]["top1_disagreements"]
            != numeric[0]["measurements"]["selected_top1_disagreements"]
            or comparison["tuned"]["top1_disagreements"]
                != numeric[1]["measurements"]["selected_top1_disagreements"]):
        raise ValueError("Pairwise numeric comparison does not match independent E2 observations")
    return {
        "schema": SCHEMA,
        "source_config_isolated": config["config_isolated"],
        "preconversion_request_isolated": config["preconversion_request_isolated"],
        "additional_postconversion_rf_keys":
            config["postconversion_additional_changed_key_count"],
        "real_native_csim_logs_fingerprint_rechecked": True,
        "real_native_csim_log_count": len(verified_log_hashes),
        "native_run_receipt_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
        "rf1_csim_result_sha256": numeric[0]["csim_result_file_sha256"],
        "rf2_csim_result_sha256": numeric[1]["csim_result_file_sha256"],
        "selected_numeric_policy_max_abs_error": 0.05,
        "rf1_selected_policy_passed":
            numeric[0]["measurements"]["selected_numeric_policy_passed"],
        "rf2_selected_policy_passed":
            numeric[1]["measurements"]["selected_numeric_policy_passed"],
        "metrics": comparison,
        "native_csim_observed_in_source_attributed_logs": True,
        "qualified_external_numeric_e2": False,
        "qualified_isolated_reuse_hardware_comparison": False,
        "native_e3_e4_or_routed_qor_observed": False,
        "vendor_tools_executed_by_replay": False,
        "full_10000_fixed_point_accuracy_measured": False,
    }


def export_native_pair_audit(stage_root, global_default, native_run_dir, output):
    dest = Path(output)
    if dest.is_symlink() or dest.exists():
        raise FileExistsError("Native replay audit requires a fresh private directory")
    record = audit_native_pair(stage_root, global_default, native_run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-native-pair-reaudit-", dir=dest.parent) as tmp:
        stage = Path(tmp) / "audit"
        stage.mkdir()
        (stage / "native_pair_reaudit_private.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        if dest.is_symlink() or dest.exists():
            raise FileExistsError("Native audit destination appeared during export")
        stage.rename(dest)
    return record
