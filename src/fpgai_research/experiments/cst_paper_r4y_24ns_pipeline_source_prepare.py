"""R4y: prospective 24 ns matched 2x2 SOURCE PACKAGES, zero native calls.

This is a NEW source-only cohort, not a reinterpretation of original
12 ns R4k/R4l or the completed R4w 16/20/24 ns routed RTL. Exactly one
intervention requests HLS PIPELINE II=1 in the producer MAC loop.
A pragma is NOT evidence of actual DSP pipelining, II or timing closure.
E2 numerical, E3 synthesis, E4 physical and board results are UNKNOWN.

Prepare is create-only, audit is strictly read-only, never executes GCC,
Vitis, Vivado or hardware. All originals are independently audited first.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp,
)
from fpgai_research.codegen.hls.r4k_full_storage_row_schedule_cpp import (
    emit_full_storage_row_schedule_cpp,
)
from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4k_matched_row_schedule_storage as r4k
from fpgai_research.experiments import cst_r4l_schedule_matched_storage_native as r4l
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package

SCHEMA = "fpgai-research/R4y-prospective-24ns-four-arm-source-package/v1"
CLOCK_NS = 24.0
UNCERTAINTY_NS = 1.5
PART = "xck26-sfvc784-2LV-c"
STORAGE = ("full_storage", "row_storage")
PIPELINE = ("baseline", "producer_mac_ii1_requested")
CASES = tuple(f"{storage}__{pipeline}"
              for storage in STORAGE for pipeline in PIPELINE)
DIRECTIVE = "      #pragma HLS PIPELINE II=1\n"
FILES = ("source.cpp", "source_manifest.json",
         "vitis_source/model.cpp", "vitis_source/tb.cpp",
         "vitis_source/run_hls.tcl", "vitis_source/manifest.json")


def _sha(data: bytes | str) -> str:
    return sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_000_000:
        raise ValueError(f"R4y unsafe/missing source receipt: {path}")
    answer = json.loads(path.read_text())
    if not isinstance(answer, dict):
        raise ValueError("R4y source receipt must be a dictionary")
    return answer


def _arch():
    graph = r4a.CASES[r4d.CASE_INDEX]
    return graph, r4a._case(graph, precision_bits=r4k.FORMAT[0],
                            integer_bits=r4k.FORMAT[1])


def _pipeline_request(emission: Emission, *, k: int) -> Emission:
    """Fail-closed one-loop pragma; no arithmetic or source event edits."""
    source = emission.cpp
    marker = f"      for(int kk=0;kk<{k};++kk) {{\n"
    if source.count(marker) != 1 or DIRECTIVE in source:
        raise ValueError("R4y canonical producer MAC loop changed")
    modified = source.replace(marker, marker + DIRECTIVE, 1)
    if modified.replace(DIRECTIVE, "", 1) != source:
        raise ValueError("R4y pipeline intervention is not one pragma")
    return Emission(modified, {
        **emission.manifest,
        "source_cpp_sha256": _sha(modified),
        "pre_intervention_source_cpp_sha256": _sha(source),
        "requested_hls_directive": "producer_mac_loop_PIPELINE_II_1",
        "requested_hls_directive_count": 1,
        "actual_dsp_registers_verified": None,
        "actual_hls_ii_verified": None,
        "E2_qualified": None,
        "E3_qualified": None,
        "E4_qualified": None,
    })


def _emissions():
    graph, arch = _arch()
    if r4k.plan()["numeric_format"] != list(r4k.FORMAT):
        raise ValueError("R4y fixed numeric source format changed")
    bases = {
        "full_storage": emit_full_storage_row_schedule_cpp(arch),
        "row_storage": emit_row_retained_fanout_cpp(arch),
    }
    source_event = {e.manifest["source_event_order_sha256"]
                    for e in bases.values()}
    functional = {e.manifest["functional_sha256"] for e in bases.values()}
    architecture = {e.manifest["architecture_sha256"] for e in bases.values()}
    if any(len(items) != 1 for items in (source_event, functional, architecture)):
        raise ValueError("R4y two storage arms differ beyond storage")
    k = graph[1][1]
    cases = {}
    for storage, base in bases.items():
        if base.manifest["source_cpp_sha256"] != _sha(base.cpp):
            raise ValueError("R4y canonical emitted C++ source mismatch")
        for pipeline in PIPELINE:
            e = base if pipeline == "baseline" else _pipeline_request(base, k=k)
            cases[f"{storage}__{pipeline}"] = e
    if set(cases) != set(CASES):
        raise ValueError("R4y prospective source factorial incomplete")
    return arch, graph, cases


def plan() -> dict:
    arch, graph, cases = _emissions()
    return {
        "schema": SCHEMA,
        "status": "PROSPECTIVE_SOURCE_ONLY_NOT_EXECUTED",
        "part": PART,
        "clock_period_ns": CLOCK_NS,
        "clock_uncertainty_ns": UNCERTAINTY_NS,
        "numeric_format": list(r4k.FORMAT),
        "graph": [graph[0], list(graph[1]), graph[2]],
        "cases": list(CASES),
        "sources_sha256": {k: _sha(e.cpp) for k, e in cases.items()},
        "source_event_order_sha256": cases[CASES[0]].manifest[
            "source_event_order_sha256"],
        "functional_sha256": cases[CASES[0]].manifest["functional_sha256"],
        "architecture_sha256": cases[CASES[0]].manifest["architecture_sha256"],
        "pipeline_request": "single producer MAC reduction loop II=1",
        "pipeline_request_effect_on_RTL": None,
        "native_vendor_calls": 0,
        "E1_GCC_qualified": False,
        "E2_CSim_qualified": False,
        "E3_HLS_qualified": False,
        "E4_routed_qualified": False,
        "board_qualified": False,
    }


def _guard_roots(output: Path, original_roots: tuple[Path, ...]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("R4y destination must not exist (create-only)")
    target = output.resolve()
    for root in original_roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("R4y original evidence root missing/symlinked")
        orig = root.resolve()
        if target == orig or orig in target.parents or target in orig.parents:
            raise ValueError("R4y cannot overlap original evidence")


def _original(*, r4k_study: str | Path,
              original_r4d: str | Path, original_r4a: str | Path) -> dict:
    audit = r4l.audit_native(study=r4k_study, original_r4d=original_r4d,
                            original_r4a=original_r4a)
    if (audit.get("source_only_E1_verified") is not True
            or audit.get("E1_matched_pair_accepted") is not True
            or audit.get("both_E2_qualified") is not True
            or audit.get("both_E3_qualified") is not True
            or audit.get("source_revision") is None
            or audit.get("new_vendor_calls_during_read_only_audit") != 0
            or audit.get("original_sources_modified") is not False):
        raise ValueError("R4y original frozen source/E2/E3 preflight rejected")
    return audit


def prepare(*, output: str | Path, r4k_study: str | Path,
            original_r4d: str | Path, original_r4a: str | Path) -> dict:
    original_roots = tuple(map(Path, (r4k_study, original_r4d, original_r4a)))
    dest = Path(output)
    _guard_roots(dest, original_roots)
    original = _original(r4k_study=r4k_study, original_r4d=original_r4d,
                         original_r4a=original_r4a)
    expected = plan()
    arch, graph, emissions = _emissions()
    samples = r4a._sample(graph[1][0], graph[1][1])
    dest.mkdir(parents=True, exist_ok=False)
    try:
        bench_sha = set()
        tcl_sha = set()
        for case in CASES:
            base = dest / case
            base.mkdir()
            emission = emissions[case]
            (base / "source.cpp").write_text(emission.cpp)
            (base / "source_manifest.json").write_text(
                json.dumps(emission.manifest, indent=2, sort_keys=True,
                           allow_nan=False) + "\n")
            pkg = base / "vitis_source"
            receipt = write_vitis_package(
                str(pkg), arch.functional, r4a._vitis_emission(arch, emission),
                samples, part=PART, clock_ns=CLOCK_NS,
                clock_uncertainty_ns=UNCERTAINTY_NS,
                tolerance=r4a.CSIM_ABS_TOLERANCE)
            if (_sha((pkg / "model.cpp").read_bytes()) !=
                    expected["sources_sha256"][case]
                    or receipt["clock_ns"] != CLOCK_NS
                    or receipt["explicit_clock_uncertainty_ns"] != UNCERTAINTY_NS
                    or receipt["part"] != PART):
                raise ValueError("R4y new source/target package not matched")
            bench_sha.add(_sha((pkg / "tb.cpp").read_bytes()))
            tcl_sha.add(_sha((pkg / "run_hls.tcl").read_bytes()))
        if len(bench_sha) != 1 or len(tcl_sha) != 1:
            raise ValueError("R4y mismatched CSim oracle or clock Tcl across arms")
        summary = {
            **expected,
            "status": "PREPARED_FOUR_SOURCE_ONLY_NEW_VITIS_PACKAGES",
            "original_R4k_source_revision": original["source_revision"],
            "original_r4k_study": str(r4k_study),
            "original_r4d": str(original_r4d),
            "original_r4a": str(original_r4a),
            "shared_testbench_sha256": next(iter(bench_sha)),
            "shared_tcl_sha256": next(iter(tcl_sha)),
            "source_revision": r4a._git_revision(),
            "originals_modified": False,
        }
        (dest / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        inventory = {p.relative_to(dest).as_posix(): _sha(p.read_bytes())
                     for p in sorted(dest.rglob("*")) if p.is_file()}
        (dest / "artifact_sha256.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        return summary
    except BaseException as exc:
        (dest / "FAILED.json").write_text(json.dumps({
            "schema": SCHEMA + "/preserved-source-preparation-failure",
            "error": type(exc).__name__ + ": " + str(exc)[:512],
            "vendor_calls": 0,
            "old_evidence_modified": False,
        }, indent=2, sort_keys=True) + "\n")
        raise


def audit(*, study: str | Path, r4k_study: str | Path,
          original_r4d: str | Path, original_r4a: str | Path) -> dict:
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4y missing or symlinked prepared source root")
    for other in (r4k_study, original_r4d, original_r4a):
        o = Path(other).resolve()
        if root.resolve() == o or root.resolve() in o.parents or o in root.resolve().parents:
            raise ValueError("R4y input overlaps frozen evidence")
    original = _original(r4k_study=r4k_study, original_r4d=original_r4d,
                         original_r4a=original_r4a)
    expected = plan()
    summary = _read(root / "summary.json")
    for k, v in expected.items():
        if summary.get(k) != v:
            raise ValueError("R4y prepared source plan changed: " + k)
    if (summary.get("status") != "PREPARED_FOUR_SOURCE_ONLY_NEW_VITIS_PACKAGES"
            or summary.get("original_R4k_source_revision") != original["source_revision"]
            or summary.get("original_r4k_study") != str(r4k_study)
            or summary.get("original_r4d") != str(original_r4d)
            or summary.get("original_r4a") != str(original_r4a)
            or summary.get("originals_modified") is not False
            or not isinstance(summary.get("source_revision"), str)
            or len(summary["source_revision"]) != 40):
        raise ValueError("R4y original provenance / prepared root mismatch")
    _, _, emission = _emissions()
    expected_files = {"summary.json"}
    for case in CASES:
        folder = root / case
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R4y source case folder missing/symlink")
        expected_files.update(f"{case}/{name}" for name in FILES)
        a, b = folder / "source.cpp", folder / "vitis_source" / "model.cpp"
        if (_sha(a.read_bytes()) != expected["sources_sha256"][case]
                or _sha(b.read_bytes()) != expected["sources_sha256"][case]
                or a.read_text() != emission[case].cpp):
            raise ValueError("R4y source or package model drift")
        manifest = _read(folder / "source_manifest.json")
        package = _read(folder / "vitis_source" / "manifest.json")
        if (manifest != emission[case].manifest
                or package.get("source_cpp_sha256") != expected["sources_sha256"][case]
                or package.get("clock_ns") != CLOCK_NS
                or package.get("explicit_clock_uncertainty_ns") != UNCERTAINTY_NS
                or package.get("testbench_sha256") != summary["shared_testbench_sha256"]):
            raise ValueError("R4y source/target/testbench manifest drift")
        if _sha((folder / "vitis_source" / "tb.cpp").read_bytes()) != summary[
            "shared_testbench_sha256"]:
            raise ValueError("R4y numerical testbench changed")
        if _sha((folder / "vitis_source" / "run_hls.tcl").read_bytes()) != summary[
            "shared_tcl_sha256"]:
            raise ValueError("R4y clock/uncertainty Tcl changed")
    expected_files.update(f"{case}/{name}" for case in CASES for name in FILES)
    manifest = _read(root / "artifact_sha256.json")
    if set(manifest) != expected_files:
        raise ValueError("R4y prepared source inventory incomplete")
    for name, digest in manifest.items():
        p = root / name
        if p.is_symlink() or not p.is_file() or _sha(p.read_bytes()) != digest:
            raise ValueError("R4y SHA-locked source file mismatch")
    all_files = {p.relative_to(root).as_posix() for p in root.rglob("*")
                 if p.is_file()}
    if all_files != expected_files | {"artifact_sha256.json"}:
        raise ValueError("R4y unexpected file in prepared source study")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_FOUR_ARM_PACKAGE_AUDITED_NOT_NATIVE",
        "source_revision": summary["source_revision"],
        "original_R4k_source_revision": original["source_revision"],
        "clock_period_ns": CLOCK_NS,
        "cases": list(CASES),
        "source_sha256": expected["sources_sha256"],
        "matched_testbench_and_tcl": True,
        "originals_modified": False,
        "new_vendor_calls_during_audit": 0,
        "E1_GCC": None,
        "E2_CSim": None,
        "E3_CSynth": None,
        "E4_Vivado": None,
        "board_qualified": False,
        "achieved_latency_ns": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--prepare-output")
    mode.add_argument("--audit-study")
    parser.add_argument("--r4k-study")
    parser.add_argument("--original-r4d")
    parser.add_argument("--original-r4a")
    args = parser.parse_args(argv)
    if args.plan:
        result = plan()
    else:
        if not all((args.r4k_study, args.original_r4d, args.original_r4a)):
            parser.error("R4y original source paths required for prepare/audit")
        kw = dict(r4k_study=args.r4k_study, original_r4d=args.original_r4d,
                  original_r4a=args.original_r4a)
        result = (prepare(output=args.prepare_output, **kw)
                  if args.prepare_output else audit(study=args.audit_study, **kw))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
