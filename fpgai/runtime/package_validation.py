from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _check(name: str, passed: bool, *, expected: Any = True, actual: Any = None, evidence: str = "", severity: str = "error") -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if bool(passed) else "failed",
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
        "severity": severity,
    }


def _file_present(package_dir: Path, relpath: str) -> tuple[bool, int | None]:
    path = package_dir / relpath
    if path.exists() and path.is_file():
        return True, path.stat().st_size
    return False, None


def _manifest_file_entries(manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files", {})
    return dict(files) if isinstance(files, Mapping) else {}


def _flatten_file_entries(files: Mapping[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for key, value in files.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("key", key)
                    flat.append(row)
        elif isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("key", key)
            flat.append(row)
    return flat


def _commands_from_sequence(payload: Mapping[str, Any]) -> list[str]:
    sequence = payload.get("sequence", [])
    commands: list[str] = []
    if isinstance(sequence, list):
        for item in sequence:
            if isinstance(item, Mapping):
                command = item.get("command")
            else:
                command = item
            if command:
                commands.append(str(command))
    return commands


def _buffer_names(buffer_plan: Mapping[str, Any]) -> set[str]:
    buffers = buffer_plan.get("buffers", [])
    names: set[str] = set()
    if isinstance(buffers, list):
        for entry in buffers:
            if isinstance(entry, Mapping) and entry.get("name"):
                names.add(str(entry["name"]))
    return names


_BUFFER_ALIAS_GROUPS: tuple[set[str], ...] = (
    {"input", "inputs", "input_mem", "inputs_mem"},
    {"output", "outputs", "output_mem", "outputs_mem"},
    {"label", "labels", "label_mem", "labels_mem"},
    {"gradient", "gradients", "gradient_mem", "gradients_mem"},
    {"weight", "weights", "weight_mem", "weights_mem"},
    {"optimizer_state", "optimizer_state_mem", "optimizer", "optimizer_mem"},
)


def _resolve_buffer_reference(name: str, available: set[str]) -> str | None:
    if name in available:
        return name
    for group in _BUFFER_ALIAS_GROUPS:
        if name in group:
            matches = sorted(group.intersection(available))
            if matches:
                return matches[0]
    return None


def _validate_required_files(package_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = [
        "package_manifest.json",
        "README_RUNTIME.md",
        "runtime_api.py",
        "board_runtime.py",
        "buffer_plan.json",
        "runtime_execution_plan.json",
    ]
    runtime_sequence = manifest.get("runtime_sequence", {})
    if isinstance(runtime_sequence, Mapping) and runtime_sequence.get("sequence"):
        required.append("run_sequence.json")
    for relpath in required:
        present, size = _file_present(package_dir, relpath)
        checks.append(
            _check(
                f"required_file_present:{relpath}",
                present,
                actual={"present": present, "bytes": size},
                evidence=relpath,
            )
        )

    files = _manifest_file_entries(manifest)
    for row in _flatten_file_entries(files):
        relpath = row.get("package_path")
        if not isinstance(relpath, str) or not relpath:
            continue
        present, size = _file_present(package_dir, relpath)
        expected_bytes = row.get("bytes")
        size_ok = present and (expected_bytes is None or int(expected_bytes) == int(size if size is not None else -1))
        checks.append(
            _check(
                f"manifest_file_entry_present:{row.get('key', relpath)}",
                size_ok,
                expected={"present": True, "bytes": expected_bytes},
                actual={"present": present, "bytes": size},
                evidence=relpath,
            )
        )
    return checks


def _validate_hardware(package_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    hardware = manifest.get("hardware", {}) if isinstance(manifest.get("hardware"), Mapping) else {}
    stages = manifest.get("build_stages", {}) if isinstance(manifest.get("build_stages"), Mapping) else {}
    bitstream_requested = bool(stages.get("bitstream"))
    vivado_impl_requested = bool(stages.get("vivado_implementation"))
    bit = hardware.get("bitstream", {}) if isinstance(hardware.get("bitstream"), Mapping) else {}
    hwh = hardware.get("hwh", {}) if isinstance(hardware.get("hwh"), Mapping) else {}
    xsa = hardware.get("xsa", {}) if isinstance(hardware.get("xsa"), Mapping) else {}
    deployable = bool(hardware.get("deployable_overlay_present"))
    bit_present = bool(bit.get("present"))
    hwh_present = bool(hwh.get("present"))
    xsa_present = bool(xsa.get("present"))
    checks.append(_check("hardware_bitstream_status_matches_request", (not bitstream_requested) or bit_present, expected=bitstream_requested, actual=bit_present, evidence="hardware.bitstream.present"))
    checks.append(_check("hardware_xsa_status_matches_request", (not bitstream_requested) or xsa_present, expected=bitstream_requested, actual=xsa_present, evidence="hardware.xsa.present"))
    checks.append(_check("hardware_deployable_overlay_consistent", deployable == (bit_present and (hwh_present or xsa_present)), expected=bit_present and (hwh_present or xsa_present), actual=deployable, evidence="hardware.deployable_overlay_present"))
    if bitstream_requested:
        checks.append(_check("hardware_package_contains_bitstream_file", (package_dir / "hardware").exists() and any((package_dir / "hardware").glob("*.bit")), evidence="runtime_package/hardware/*.bit"))
        checks.append(_check("hardware_package_contains_handoff_file", (package_dir / "hardware").exists() and (any((package_dir / "hardware").glob("*.hwh")) or any((package_dir / "hardware").glob("*.xsa"))), evidence="runtime_package/hardware/*.hwh or *.xsa"))
    else:
        checks.append(_check("hardware_bitstream_not_required_for_impl_only", not bit_present and not xsa_present if vivado_impl_requested else True, expected=False, actual={"bitstream_present": bit_present, "xsa_present": xsa_present}, evidence="implementation-only package should not include stale bitstream/xsa"))
    return checks


def _validate_buffers_and_sequence(package_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    buffer_plan = _read_json(package_dir / "buffer_plan.json")
    execution_plan = _read_json(package_dir / "runtime_execution_plan.json")
    run_sequence = _read_json(package_dir / "run_sequence.json")
    if not run_sequence:
        run_sequence = manifest.get("runtime_sequence", {}) if isinstance(manifest.get("runtime_sequence"), Mapping) else {}
    names = _buffer_names(buffer_plan)
    resolved_input = _resolve_buffer_reference("input", names)
    resolved_output = _resolve_buffer_reference("output", names)
    checks.append(
        _check(
            "buffer_plan_has_input",
            resolved_input is not None,
            expected="input/input_mem alias group",
            actual={"available_buffers": sorted(names), "resolved": resolved_input},
            evidence="buffer_plan.json; generated m_axi packages may expose input_mem while logical runtime uses input",
        )
    )
    checks.append(
        _check(
            "buffer_plan_has_output",
            resolved_output is not None,
            expected="output/output_mem alias group",
            actual={"available_buffers": sorted(names), "resolved": resolved_output},
            evidence="buffer_plan.json; generated m_axi packages may expose output_mem while logical runtime uses output",
        )
    )

    buffers = buffer_plan.get("buffers", [])
    if isinstance(buffers, list):
        for entry in buffers:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or "<missing>")
            words = entry.get("words")
            nbytes = entry.get("bytes")
            direction = str(entry.get("direction") or "")
            valid = bool(entry.get("name")) and direction in {"ps_to_pl", "pl_to_ps", "bidirectional"} and int(words or 0) >= 1 and int(nbytes or 0) >= 4
            checks.append(_check(f"buffer_entry_valid:{name}", valid, actual={"direction": direction, "words": words, "bytes": nbytes}, evidence="buffer_plan.json"))

    commands = _commands_from_sequence(run_sequence)
    seq = execution_plan.get("sequence", [])
    actual_commands = [str(item.get("command")) for item in seq if isinstance(item, Mapping) and item.get("command")] if isinstance(seq, list) else []
    checks.append(_check("runtime_execution_plan_matches_run_sequence", actual_commands == commands, expected=commands, actual=actual_commands, evidence="runtime_execution_plan.json/run_sequence.json"))

    if "export_gradients" in commands:
        resolved_gradients = _resolve_buffer_reference("gradients", names)
        checks.append(
            _check(
                "buffer_plan_has_gradient_export_buffer",
                resolved_gradients is not None,
                expected="gradients/gradients_mem alias group",
                actual={"available_buffers": sorted(names), "resolved": resolved_gradients},
                evidence="export_gradients requires a gradient export buffer; generated packages may use gradients_mem",
            )
        )
    if "export_optimizer_state" in commands:
        resolved_optimizer = _resolve_buffer_reference("optimizer_state", names)
        checks.append(
            _check(
                "buffer_plan_has_optimizer_state_buffer",
                resolved_optimizer is not None,
                expected="optimizer_state/optimizer_state_mem alias group",
                actual={"available_buffers": sorted(names), "resolved": resolved_optimizer},
                evidence="export_optimizer_state requires an optimizer-state export buffer; generated packages may use optimizer_state_mem",
            )
        )
    if any(command in {"run_training", "accumulate_gradients"} for command in commands):
        label_candidates = {"labels", "label", "label_mem", "labels_mem"}
        present_labels = sorted(name for name in names if name in label_candidates)
        checks.append(
            _check(
                "buffer_plan_has_training_label_buffer",
                bool(present_labels),
                expected=sorted(label_candidates),
                actual={"available_buffers": sorted(names), "matched_label_buffers": present_labels},
                evidence="training commands require a label buffer; generated m_axi training packages use label_mem",
            )
        )

    if isinstance(seq, list):
        for idx, item in enumerate(seq):
            if not isinstance(item, Mapping):
                continue
            refs = [str(name) for name in (list(item.get("sync_before", []) or []) + list(item.get("sync_after", []) or []))]
            resolved = {name: _resolve_buffer_reference(name, names) for name in refs}
            missing = [name for name, match in resolved.items() if match is None]
            checks.append(
                _check(
                    f"execution_step_buffers_resolve:{idx}:{item.get('command')}",
                    not missing,
                    expected="all sync buffers in buffer_plan",
                    actual={"missing": missing, "resolved": resolved, "available_buffers": sorted(names)},
                    evidence="runtime_execution_plan.json; common buffer aliases are accepted for generated m_axi training ABI",
                )
            )
    return checks


def _validate_runtime_api(package_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    api_path = package_dir / "runtime_api.py"
    board_path = package_dir / "board_runtime.py"
    api = api_path.read_text(encoding="utf-8", errors="replace") if api_path.exists() else ""
    board = board_path.read_text(encoding="utf-8", errors="replace") if board_path.exists() else ""
    required_api = ["load_manifest", "load_buffer_plan", "allocate_runtime_buffers", "bind_backend", "run_sequence"]
    for name in required_api:
        checks.append(_check(f"runtime_api_function_present:{name}", f"def {name}" in api, evidence="runtime_api.py"))
    required_backend = ["FPGAIBoardRuntime", "PynqDmaMmioBackend", "create_pynq_backend", "FPGAI_MODE_RUN_TRAINING"]
    for name in required_backend:
        checks.append(_check(f"board_runtime_symbol_present:{name}", name in board, evidence="board_runtime.py"))
    functions = manifest.get("runtime_api", {}).get("functions", []) if isinstance(manifest.get("runtime_api"), Mapping) else []
    for name in functions if isinstance(functions, list) else []:
        checks.append(_check(f"runtime_api_manifest_function_present:{name}", f"def {name}" in api or str(name) in api, evidence="runtime_api.functions/runtime_api.py", severity="warning"))
    return checks


def emit_runtime_package_validation(out_dir: str | Path, package_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate runtime package deployability metadata without running hardware.

    The report is a static package contract: it verifies that the generated runtime
    package is internally consistent and ready to be copied to a target board when
    bitstream artifacts are present. It does not claim board execution.
    """
    root = Path(out_dir).resolve()
    pkg = Path(package_dir).resolve() if package_dir is not None else root / "runtime_package"
    manifest_path = pkg / "package_manifest.json"
    manifest = _read_json(manifest_path)
    checks: dict[str, list[dict[str, Any]]] = {
        "package_files": [_check("package_manifest_readable", bool(manifest), actual=manifest_path.exists(), evidence=_rel(manifest_path, root))],
        "hardware": [],
        "buffers_and_sequence": [],
        "runtime_api": [],
    }
    if manifest:
        checks["package_files"].extend(_validate_required_files(pkg, manifest))
        checks["hardware"].extend(_validate_hardware(pkg, manifest))
        checks["buffers_and_sequence"].extend(_validate_buffers_and_sequence(pkg, manifest))
        checks["runtime_api"].extend(_validate_runtime_api(pkg, manifest))

    flat = [row for group in checks.values() for row in group]
    failed = [row for row in flat if row.get("status") != "passed" and row.get("severity") != "warning"]
    warnings = [row for row in flat if row.get("status") != "passed" and row.get("severity") == "warning"]
    hardware = manifest.get("hardware", {}) if isinstance(manifest.get("hardware"), Mapping) else {}
    build_stages = manifest.get("build_stages", {}) if isinstance(manifest.get("build_stages"), Mapping) else {}
    deployable = bool(hardware.get("deployable_overlay_present"))
    bitstream_requested = bool(build_stages.get("bitstream"))
    failed_summary = [
        {
            "group": group,
            "name": str(row.get("name", "")),
            "expected": row.get("expected"),
            "actual": row.get("actual"),
            "evidence": row.get("evidence", ""),
        }
        for group, rows in checks.items()
        for row in rows
        if row.get("status") != "passed" and row.get("severity") != "warning"
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "package_kind": "fpgai_runtime_package_validation",
        "status": "passed" if not failed else "failed",
        "deployability_ready": bool(deployable and not failed),
        "board_execution_claimed": False,
        "bitstream_requested": bitstream_requested,
        "deployable_overlay_present": deployable,
        "check_count": len(flat),
        "failed_count": len(failed),
        "warning_count": len(warnings),
        "failed_checks": failed_summary,
        "checks": checks,
        "paths": {
            "out_dir": root.as_posix(),
            "package_dir": pkg.as_posix(),
            "package_manifest": manifest_path.as_posix(),
        },
        "truth_boundary": "Static runtime-package validation only. It validates files, metadata, buffer plans, and generated runtime APIs; it does not execute on the FPGA board.",
    }

    pkg_report_json = pkg / "runtime_package_validation.json"
    pkg_report_md = pkg / "runtime_package_validation.md"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    root_report_json = reports_dir / "runtime_package_validation.json"
    root_report_md = reports_dir / "runtime_package_validation.md"

    text = json.dumps(report, indent=2, sort_keys=True)
    pkg_report_json.write_text(text, encoding="utf-8")
    root_report_json.write_text(text, encoding="utf-8")

    lines = [
        "# Runtime Package Validation",
        "",
        f"- status: `{report['status']}`",
        f"- deployability_ready: `{report['deployability_ready']}`",
        f"- board_execution_claimed: `{report['board_execution_claimed']}`",
        f"- bitstream_requested: `{report['bitstream_requested']}`",
        f"- deployable_overlay_present: `{report['deployable_overlay_present']}`",
        f"- check_count: `{report['check_count']}`",
        f"- failed_count: `{report['failed_count']}`",
        f"- warning_count: `{report['warning_count']}`",
        f"- failed_checks: `{json.dumps(report['failed_checks'], sort_keys=True)}`",
        "",
        "## Checks",
    ]
    for group, rows in checks.items():
        lines.extend(["", f"### {group}", ""])
        for row in rows:
            lines.append(f"- **{row['status'].upper()}** {row['name']}: expected=`{row.get('expected')}` actual=`{row.get('actual')}` evidence=`{row.get('evidence', '')}`")
    lines.extend(["", "## Boundary", "", str(report["truth_boundary"]), ""])
    md = "\n".join(lines)
    pkg_report_md.write_text(md, encoding="utf-8")
    root_report_md.write_text(md, encoding="utf-8")

    return {
        "status": report["status"],
        "deployability_ready": report["deployability_ready"],
        "board_execution_claimed": False,
        "validation_json": "runtime_package/runtime_package_validation.json",
        "validation_md": "runtime_package/runtime_package_validation.md",
        "reports_json": "reports/runtime_package_validation.json",
        "reports_md": "reports/runtime_package_validation.md",
        "check_count": report["check_count"],
        "failed_count": report["failed_count"],
        "warning_count": report["warning_count"],
    }
