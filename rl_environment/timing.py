"""Strict original Vitis HLS estimated-clock gate; no routed/physical claims.

Only a freshly attributed csynth XML from the original invocation is eligible.
TargetClockPeriod and EstimatedClockPeriod are Vitis *HLS* estimates, not Fmax.
"""
from __future__ import annotations

from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
from xml.etree import ElementTree as ET


XML_REL = Path("fpgai_hls_project/solution1/syn/report/fpgai_model_csynth.xml")


def _unique_period(parent, tag, *, allow_zero=False):
    values = parent.findall(tag)
    if len(values) != 1 or values[0].text is None:
        return None
    try:
        value = float(values[0].text.strip())
    except ValueError:
        return None
    return value if isfinite(value) and (value >= 0 if allow_zero else value > 0) else None


def parse_hls_timing(xml_path, requested_clock_ns, *,
                     source_attested_uncertainty_ns=None):
    """Fail closed on missing, ambiguous or contradictory HLS timing evidence.

    Does not infer timing from cycles, Tcl alone or an arbitrary log string.
    A tool/version-specific report must actually contain these known XML tags.
    """
    if not isfinite(float(requested_clock_ns)) or requested_clock_ns <= 0:
        raise ValueError("Invalid requested HLS clock")
    xml = Path(xml_path)
    if xml.is_symlink() or not xml.is_file():
        raise ValueError("Missing or symlinked original HLS XML")
    root = ET.parse(xml).getroot()
    timing = root.findall(".//PerformanceEstimates/SummaryOfTimingAnalysis")
    if len(timing) != 1:
        raise ValueError("Missing or ambiguous HLS timing section")
    # Vitis csynth XML commonly records the user-requested clock in
    # ./UserAssignments/TargetClockPeriod, but EstimatedClockPeriod in the
    # PerformanceEstimates/SummaryOfTimingAnalysis section. Older fixtures
    # have the requested clock in the timing section itself. Only these two
    # exact locations are supported; malformed/duplicate/conflicting data
    # must never be treated as a successful HLS timing observation.
    user_sections = root.findall("./UserAssignments")
    if len(user_sections) > 1:
        raise ValueError("Ambiguous Vitis HLS user assignments")
    user = user_sections[0] if user_sections else None
    targets = []
    for location, section in (
            ("UserAssignments", user),
            ("SummaryOfTimingAnalysis", timing[0])):
        if section is None:
            continue
        entries = section.findall("TargetClockPeriod")
        if entries:
            value = _unique_period(section, "TargetClockPeriod")
            if value is None:
                raise ValueError("Missing or malformed HLS target/estimated period")
            targets.append((location, value))
    estimate = _unique_period(timing[0], "EstimatedClockPeriod")
    if not targets or estimate is None:
        raise ValueError("Missing or malformed HLS target/estimated period")
    target = targets[0][1]
    if any(not isclose(value, target, rel_tol=0, abs_tol=0.0001)
           for _, value in targets[1:]):
        raise ValueError("Conflicting HLS target periods across XML sections")
    if not isclose(target, float(requested_clock_ns), rel_tol=0, abs_tol=0.0001):
        raise ValueError("Vitis reported a different requested clock")
    # HLS tool versions differ on clock uncertainty accounting: record a
    # valid field but do not blindly add it to EstimatedClockPeriod.
    uncertainties = []
    for section in (user, timing[0]):
        if section is None:
            continue
        if section.findall("ClockUncertainty"):
            value = _unique_period(section, "ClockUncertainty", allow_zero=True)
            if value is None or value >= target:
                raise ValueError("Invalid or conflicting HLS clock uncertainty")
            uncertainties.append(value)
    if len(uncertainties) > 1 and not isclose(
            uncertainties[0], uncertainties[1], rel_tol=0, abs_tol=0.0001):
        raise ValueError("Conflicting HLS clock uncertainty across XML sections")
    # UG1399 (Vitis HLS 2023.2) defines the effective scheduling budget
    # as target period MINUS clock uncertainty (default 27% in 2023.2).
    # EstimatedClockPeriod measures the estimated operator/state delay.
    # Comparing it directly to the unadjusted requested period incorrectly
    # accepts paths that miss HLS's effective scheduling constraint.
    # Never silently assume zero uncertainty if the evidence is missing.
    if source_attested_uncertainty_ns is not None:
        # The caller MUST verify the exact generated Tcl, invocation script
        # digest and original executed Vitis input fingerprints. This is NOT
        # a generic permission to assume a 27% default on arbitrary XML.
        source_value = source_attested_uncertainty_ns
        if (type(source_value) not in (int, float)
                or not isfinite(source_value)
                or not 0. <= source_value < target):
            raise ValueError("Invalid source-attested explicit HLS uncertainty")
        if uncertainties and not isclose(
                uncertainties[0], source_value, rel_tol=0, abs_tol=.0001):
            raise ValueError("Vitis XML and executed Tcl HLS uncertainties disagree")
    if not uncertainties and source_attested_uncertainty_ns is None:
        raise ValueError("Missing HLS clock uncertainty: cannot establish effective timing budget")
    uncertainty = (uncertainties[0] if uncertainties
                   else float(source_attested_uncertainty_ns))
    effective_budget = target - uncertainty
    if not isfinite(effective_budget) or effective_budget <= 0:
        raise ValueError("Invalid effective HLS scheduling clock budget")
    return {
        "hls_target_clock_ns": target,
        "hls_estimated_clock_ns": estimate,
        "hls_clock_uncertainty_ns": uncertainty,
        "hls_effective_scheduling_budget_ns": effective_budget,
        "hls_scheduling_margin_ns": effective_budget - estimate,
        "hls_timing_feasible": estimate <= effective_budget,
        "timing_source": ("original-vitis-csynth-xml"
                          if uncertainties else
                          "original-vitis-csynth-xml-plus-hash-attested-executed-tcl"),
        "hls_report_sha256": sha256(xml.read_bytes()).hexdigest(),
        "physical_fmax_mhz": None,
    }


def audited_hls_timing(case, native, summary, clock_ns):
    """Require exact source-attested XML receipt before parsing timing."""
    case = Path(case)
    path = case / XML_REL
    expected = (native.get("fresh_report_sha256") or {}).get("csynth_xml")
    if (summary.get("evidence") != "E3-HLS-synthesis-report"
            or not isinstance(expected, str) or not path.is_file()
            or path.is_symlink() or expected != sha256(path.read_bytes()).hexdigest()
            or summary.get("hls_report_sha256") != expected):
        raise ValueError("HLS timing report not fresh and source-attributed")
    manifest_path = case / "manifest.json"
    import json
    if manifest_path.is_symlink():
        raise ValueError("Original HLS manifest must not be a symlink")
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {})
    explicit = manifest.get("explicit_clock_uncertainty_ns")
    if explicit is not None:
        from math import isfinite as _finite
        if (type(explicit) not in (float, int)
                or not _finite(explicit) or explicit < 0
                or explicit >= clock_ns):
            raise ValueError("Invalid explicit uncertainty in original native manifest")
        script = case / "run_hls.tcl"
        if script.is_symlink() or not script.is_file():
            raise ValueError("Missing original uncertainty Tcl")
        digest = sha256(script.read_bytes()).hexdigest()
        expected = (
            native.get("script_sha256") == digest
            and native.get("input_hashes_before_execution", {}).get("tcl") == digest
            and native.get("input_hashes_after_execution", {}).get("tcl") == digest
            and native.get("inputs_stable_during_execution") is True
            and native.get("completed_successfully") is True)
        lines = script.read_text(encoding="utf-8").splitlines()
        uncertainty_commands = [
            line for line in lines if line.strip().startswith("set_clock_uncertainty")]
        exact_command = f"set_clock_uncertainty {explicit:.3f}"
        if not expected or uncertainty_commands != [exact_command]:
            raise ValueError(
                "Original source-attested Vitis clock uncertainty Tcl missing or changed")
        # Only the one explicit, executed source-attested Tcl setting may fill
        # the absent XML uncertainty. Target and estimated clock still come
        # from the original fresh and digest-matched Vitis csynth report.
    return parse_hls_timing(
        path, clock_ns, source_attested_uncertainty_ns=explicit)


def probe_case_timing_schema(case):
    """Read-only compatibility check for an *existing* HLS case.

    This deliberately does NOT qualify source integrity, numerical accuracy,
    resources or a new Gym result. Only audited_hls_timing() can qualify
    a newly executed and source-attributed original vendor case.
    """
    import json
    case = Path(case)
    if not case.is_dir() or case.is_symlink():
        raise ValueError("Expected an existing nonsymlinked HLS case directory")
    manifest_path = case / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("No trustworthy HLS manifest in selected case")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "clock_ns" not in manifest:
        raise ValueError("Case manifest does not define its requested clock")
    clock = manifest["clock_ns"]
    if type(clock) not in (int, float):
        raise ValueError("Manifest clock must be numeric")
    observed = parse_hls_timing(case / XML_REL, clock)
    return {
        "status": "READ_ONLY_XML_SCHEMA_COMPATIBLE_NOT_NATIVE_QUALIFICATION",
        "hls_target_clock_ns": observed["hls_target_clock_ns"],
        "hls_estimated_clock_ns": observed["hls_estimated_clock_ns"],
        "hls_clock_uncertainty_ns": observed["hls_clock_uncertainty_ns"],
        "hls_effective_scheduling_budget_ns": observed["hls_effective_scheduling_budget_ns"],
        "hls_scheduling_margin_ns": observed["hls_scheduling_margin_ns"],
        "schema_timing_feasible_only": observed["hls_timing_feasible"],
        "hls_report_sha256": observed["hls_report_sha256"],
        "qualified_for_new_gym_science": False,
        "source_csim_resource_attestation_performed": False,
        "native_vendor_invocations_during_probe": 0,
        "physical_fmax_mhz": None,
    }


def inspect_case_timing_schema(case):
    """Print only recognized clock-field locations and numeric values.

    This read-only diagnostic does not export arbitrary XML text, private
    model contents, C++ sources, weights, operator identities or file paths.
    It is not source/CSim/resource/timing qualification of a new study.
    """
    import json
    case = Path(case)
    if not case.is_dir() or case.is_symlink():
        raise ValueError("Expected an existing nonsymlinked HLS case directory")
    manifest_file = case / "manifest.json"
    xml_file = case / XML_REL
    if (not manifest_file.is_file() or manifest_file.is_symlink()
            or not xml_file.is_file() or xml_file.is_symlink()):
        raise ValueError("Missing existing native manifest or HLS XML")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    clock = manifest.get("clock_ns")
    if type(clock) not in (float, int) or not isfinite(clock) or clock <= 0:
        raise ValueError("Existing manifest clock must be finite positive numeric")
    root = ET.parse(xml_file).getroot()
    locations = (
        ("UserAssignments", "./UserAssignments"),
        ("SummaryOfTimingAnalysis",
         ".//PerformanceEstimates/SummaryOfTimingAnalysis"),
    )
    fields = {}
    sections = {}
    for label, xpath in locations:
        matches = root.findall(xpath)
        sections[label] = len(matches)
        for field in ("TargetClockPeriod", "EstimatedClockPeriod",
                      "ClockUncertainty"):
            entry_count = sum(len(item.findall(field)) for item in matches)
            value = (_unique_period(matches[0], field,
                                    allow_zero=field == "ClockUncertainty")
                     if len(matches) == 1 else None)
            fields[label + "/" + field] = {
                "count": entry_count,
                "numeric_ns": value,
            }
    return {
        "status": "READ_ONLY_XML_FIELD_DIAGNOSTIC_NOT_QUALIFICATION",
        "manifest_requested_clock_ns": float(clock),
        "section_counts": sections,
        "recognized_clock_fields": fields,
        "hls_report_sha256": sha256(xml_file.read_bytes()).hexdigest(),
        "qualified_for_new_gym_science": False,
        "original_vendor_invocations": 0,
        "physical_fmax_mhz": None,
    }


def main(argv=None):
    """Inspect an earlier, existing original Vitis report without running Vitis."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True,
                        help="Existing HLS case directory containing manifest.json")
    parser.add_argument("--inspect-schema", action="store_true",
                        help="Inspect recognized clock XML field locations safely; "
                             "never qualify timing or launch Vitis")
    args = parser.parse_args(argv)
    result = (inspect_case_timing_schema(args.case)
              if args.inspect_schema else probe_case_timing_schema(args.case))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
