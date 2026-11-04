"""Evidence-aware paper result collection; missing measurements remain null."""
from hashlib import sha256
from pathlib import Path
from xml.etree.ElementTree import ParseError
import json
from fpgai_research.toolchains.vitis_hls.package import parse_csynth_xml
from fpgai_research.toolchains.vitis_hls.report import discover_csim_success
from fpgai_research.toolchains.vivado.report import parse_vivado_reports
from fpgai_research.toolchains.runner import rtl_sha256

METRICS=("latency_best","latency_worst","ii_min","ii_max","lut","ff","dsp","bram_18k","uram")

def _native_tool_identity(receipt: dict | None, tool: str) -> bool:
    """Require recognizable native vendor tool identity for evidence promotion.

    A banner and program name provide basic sanity checking, not a
    cryptographic authenticity guarantee.
    """
    import re
    if receipt is None or tool not in ("vitis_hls", "vivado"):
        return False
    executable = receipt.get("executable")
    banner = receipt.get("tool_version_output")
    if not isinstance(executable, str) or not isinstance(banner, str):
        return False
    if Path(executable).name.lower() not in (tool, tool + ".exe"):
        return False
    pattern = r"\bvitis\s+hls\b" if tool == "vitis_hls" else r"\bvivado\s+v?\d"
    return re.search(pattern, banner, re.IGNORECASE) is not None


def _attested(path: Path, tool: str, script: Path, source: Path,
              testbench: Path | None = None) -> dict | None:
    if not path.is_file() or not script.is_file():
        return None
    try:
        record=json.loads(path.read_text())
    except (OSError,ValueError):
        return None
    valid=(record.get("schema")=="fpgai-research/vendor-invocation/v1"
            and record.get("tool")==tool
            and record.get("completed_successfully") is True
            and record.get("returncode")==0
            and record.get("timed_out") is False
            and record.get("script_sha256")==sha256(script.read_bytes()).hexdigest()
            and record.get("source_sha256")==sha256(source.read_bytes()).hexdigest()
            and (testbench is None or
                 (testbench.is_file() and record.get("testbench_sha256")==sha256(testbench.read_bytes()).hexdigest())))
    return record if valid else None

def _csim_log_attributed(case: Path, receipt: dict | None, sim: dict) -> bool:
    """Accept E2 logs only when the same completed invocation hashed them."""
    if receipt is None or not sim.get("marker") or not sim.get("source"):
        return False
    origin = sim["source"]
    candidates = {
        "logs/vitis.stdout.log": ("stdout_sha256", None),
        "fpgai_hls_project/solution1/csim/report/fpgai_model_csim.log": ("fresh_report_sha256", "csim_log"),
        "fpgai_hls_project/solution1/csim/build/csim_results.log": ("fresh_report_sha256", "csim_results"),
    }
    if origin not in candidates:
        return False
    field, subfield = candidates[origin]
    recorded = receipt.get(field)
    if subfield is not None:
        recorded = (recorded or {}).get(subfield)
    log_path = case / origin
    return bool(log_path.is_file() and recorded and
                recorded == sha256(log_path.read_bytes()).hexdigest())


def summarize_case(folder: str) -> dict:
    case=Path(folder)
    # Archived model/data inputs are optional, but when present their content
    # and prepared-source provenance must match before accepting any evidence.
    from fpgai_research.experiments.input_archive import verify_case_provenance
    if (case/"inputs").exists() or (case/"inputs").is_symlink():
        verify_case_provenance(case)
    elif (case.parent/"campaign.json").is_file() and ((case.parent/"inputs").exists() or (case.parent/"inputs").is_symlink()):
        verify_case_provenance(case.parent)
    manifest=case/"manifest.json"
    source=case/"model.cpp"
    if not manifest.is_file() or not source.is_file():
        raise FileNotFoundError("Case is missing manifest.json or model.cpp")
    meta=json.loads(manifest.read_text())
    actual=sha256(source.read_bytes()).hexdigest()
    if actual!=meta.get("cpp_sha256"):
        raise ValueError("Generated source hash differs from its manifest")
    artifact_file=case/"progressive_ir.json"
    if artifact_file.is_file() or artifact_file.is_symlink():
        if artifact_file.is_symlink():
            raise ValueError("Symlinked scientific IR artifact is not accepted")
        from fpgai_research.model_ir.artifacts import read_ir_artifact
        artifact=read_ir_artifact(artifact_file)
        if artifact["stages"]["hls"]["manifest"]["cpp_sha256"]!=actual:
            raise ValueError("Scientific Model IR artifact disagrees with HLS source hash")
    if meta.get("testbench_sha256") is not None:
        tb=case/"tb.cpp"
        if not tb.is_file() or sha256(tb.read_bytes()).hexdigest()!=meta["testbench_sha256"]:
            raise ValueError("Numerical testbench hash mismatch")
    row={"name":case.name,"source_sha256":actual,"evidence":"E1-HLS-source",
         "metrics":{k:None for k in METRICS},"vivado_metrics":None,
         "csim_passed":None,"warnings":[],
         "precision_bits":meta.get("precision_bits"),
         "integer_bits":meta.get("integer_bits"),
         "numerical_tolerance":meta.get("numerical_tolerance"),
         "target_part":meta.get("part"),
         "clock_ns":meta.get("clock_ns")}
    candidate=_attested(case/"vitis_invocation.json","vitis_hls",case/"run_hls.tcl",source,testbench=case/"tb.cpp")
    vitis_receipt=candidate if _native_tool_identity(candidate, "vitis_hls") else None
    if candidate is not None and vitis_receipt is None:
        row["warnings"].append("Vitis report cannot be promoted: native tool identity/version unverified")
    vitis_ok=vitis_receipt is not None
    sim=discover_csim_success(case)
    csim_attested = _csim_log_attributed(case, vitis_receipt, sim)
    # Attested numerical metrics must cover the complete prepared testbench.
    coverage_valid = (meta.get("csim_sample_count") is None or
                      sim.get("samples") == meta["csim_sample_count"]) and (
                      meta.get("csim_outputs_per_sample") is None or
                      sim.get("outputs_per_sample") == meta["csim_outputs_per_sample"])
    if csim_attested and not coverage_valid:
        row["warnings"].append("Attested C simulation coverage does not match the prepared testbench")
    if csim_attested and coverage_valid and sim["passed"]:
        row.update(evidence="E2-Vitis-C-simulation",csim_passed=True,csim_receipt=sim)
    elif sim.get("marker") and csim_attested and coverage_valid and not sim["passed"]:
        row.update(csim_passed=False, csim_receipt=sim)
        row["warnings"].append("C simulation completed but numerical acceptance failed")
    elif sim.get("marker"):
        row["warnings"].append("C-simulation marker lacks a fresh log from the same source-attested Vitis invocation")
    xml=case/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.xml"
    if xml.is_file():
        if (vitis_ok and vitis_receipt.get("fresh_report_sha256",{}).get("csynth_xml")==sha256(xml.read_bytes()).hexdigest()):
            # Tool-generated XML may contain 'undef' latency or be truncated.
            # Such a value is NOT a synthesis measurement and cannot qualify
            # E3. Preserve exact source verification and independently audited
            # cases; do not abort the entire multi-operator experiment.
            try:
                parsed=parse_csynth_xml(str(xml))
            except (ValueError,ParseError) as exc:
                row["warnings"].append(
                    f"HLS XML measurement rejected: {type(exc).__name__}: {exc}")
            else:
                row.update(evidence="E3-HLS-synthesis-report",
                           metrics=parsed["measurements"],
                           hls_report_sha256=parsed["report_sha256"])
        else:
            row["warnings"].append("HLS XML lacks a fresh matching source-attested Vitis invocation")
    vivado_receipt=_attested(case/"vivado"/"vivado_invocation.json",
                       "vivado",case/"vivado"/"run_vivado.tcl",source)
    if vivado_receipt is not None and not _native_tool_identity(vivado_receipt, "vivado"):
        row["warnings"].append("Vivado report cannot be promoted: native tool identity/version unverified")
        vivado_receipt=None
    if vivado_receipt:
        if row["evidence"]!="E3-HLS-synthesis-report":
            row["warnings"].append("Vivado finished but upstream HLS synthesis was not verified")
        elif (not vitis_receipt.get("rtl_output_sha256")
              or not vivado_receipt.get("rtl_input_sha256")
              or vitis_receipt["rtl_output_sha256"]!=vivado_receipt["rtl_input_sha256"]
              or rtl_sha256(case)!=vivado_receipt["rtl_input_sha256"]):
            row["warnings"].append("Vivado RTL inputs differ from source-attested Vitis HLS generated Verilog")
        else:
            # A native Vivado process may exit 0 even when routing or timing
            # reports are incomplete. Preserve attested E3 evidence, but never
            # promote an unparseable/partially routed report to E4.
            try:
                result=parse_vivado_reports(case/"vivado")
            except (ValueError, FileNotFoundError) as exc:
                row["warnings"].append(
                    f"Vivado E4 report rejected: {type(exc).__name__}: {exc}")
            else:
                if all(vivado_receipt.get("fresh_report_sha256",{}).get(name)==digest
                       for name,digest in result["raw_report_sha256"].items()):
                    row.update(evidence="E4-Vivado-implementation",vivado_metrics=result)
                    # The exact Tcl hash is tied to the successful Vivado
                    # invocation, so retain the implementation context for
                    # like-for-like paper comparisons. OOC is module QoR,
                    # never a deployed FPGA/board claim.
                    tcl=(case/"vivado"/"run_vivado.tcl").read_text()
                    row["vivado_implementation_mode"]=(
                        "out_of_context" if
                        "synth_design -mode out_of_context " in tcl
                        else "standalone_pin_placed")
                    drc=result["drc"]
                    if not drc["summary_verified"]:
                        row["warnings"].append(
                            "Vivado routed/timing evidence exists but DRC report is absent or unverified; not board-deployable")
                    elif drc["violations"]:
                        row["warnings"].append(
                            f'Vivado DRC contains {drc["violations"]} violations '
                            f'({drc["critical_warnings"]} critical/error); '
                            'standalone HLS top is not qualified for board deployment')
                else:
                    row["warnings"].append("Vivado reports not fresh and bound to this invocation")
    elif (case/"vivado"/"reports"/"route_status.rpt").is_file():
        row["warnings"].append("Vivado reports without matching completed Vivado invocation")
    # A failed Vivado invocation is not a physical-evidence result, but its
    # failure must not be silently omitted from an otherwise valid E3 record.
    # Attribute this diagnostic to the exact prepared source and Tcl hash;
    # arbitrary or stale receipts are not trusted as current run failures.
    failed_vivado=case/"vivado"/"vivado_invocation.json"
    if failed_vivado.is_file() and not failed_vivado.is_symlink():
        try:
            failed=json.loads(failed_vivado.read_text())
        except (OSError, ValueError):
            failed=None
        script=case/"vivado"/"run_vivado.tcl"
        if (isinstance(failed,dict)
            and failed.get("schema")=="fpgai-research/vendor-invocation/v1"
            and failed.get("tool")=="vivado"
            and failed.get("completed_successfully") is False
            and failed.get("source_sha256")==actual
            and script.is_file()
            and failed.get("script_sha256")==sha256(script.read_bytes()).hexdigest()):
            exitcode=failed.get("returncode")
            suffix=("timeout" if failed.get("timed_out") is True
                    else f"returncode={exitcode}" if type(exitcode) is int
                    else "unknown exit status")
            row["warnings"].append(
                "Vivado native invocation failed ("+suffix+
                "); E4 not established. Inspect vivado/logs/vivado.stdout.log "
                "and vivado/logs/vivado.stderr.log")
    host=case/"host_validation.json"
    if host.is_file():
        item=json.loads(host.read_text())
        if item.get("cpp_sha256")!=actual or item.get("mode")!="gcc-host-float":
            raise ValueError("Host float record does not match this source")
        row["host_float_check"]={k:item.get(k) for k in ("accepted","rmse","max_absolute_error","cosine")}
    return row

def export_results(directories: list[str], destination: str) -> list[dict]:
    rows=[summarize_case(path) for path in directories]
    Path(destination).write_text(json.dumps({
       "schema":"fpgai-research/paper-evidence/v2",
       "unreported_values":None,"results":rows},indent=2,sort_keys=True)+"\n")
    return rows
