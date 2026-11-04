"""Explicit researcher-controlled prepare/run/collect workflow."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.model_ir.lowering import lower
from fpgai_research.architecture.resolver import resolve, TargetLimits
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from fpgai_research.toolchains.vitis_hls.runner import run_vitis
from fpgai_research.toolchains.vivado.project import write_vivado_package
from fpgai_research.toolchains.vivado.runner import run_vivado
from fpgai_research.experiments.results import export_results
from fpgai_research.experiments.campaign import prepare_campaign
from fpgai_research.experiments.paper_tables import export_tables
from fpgai_research.toolchains.staging import staged_experiment
from fpgai_research.cli import _request

def main(argv=None):
    ap=argparse.ArgumentParser(description="FPGAI research: explicit paper reproduction stages")
    sub=ap.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect-model",help="List exact ONNX operator IDs for paper campaigns")
    inspect.add_argument("--onnx",required=True)
    reference=sub.add_parser("reference-eval",help="Save NumPy Model IR outputs and labeled accuracy; E0 only")
    reference.add_argument("--onnx",required=True)
    reference.add_argument("--input",required=True)
    reference.add_argument("--labels")
    reference.add_argument("--output",required=True)
    reference.add_argument("--compare-onnxruntime",action="store_true")
    run_campaign_parser=sub.add_parser("run-campaign",help="Opt-in bounded Vitis/Vivado campaign execution")
    run_campaign_parser.add_argument("--directory",required=True)
    run_campaign_parser.add_argument("--stage",action="append",choices=("vitis","vivado"),required=True)
    run_campaign_parser.add_argument("--execute",action="store_true",help="Actually invoke local licensed vendor tools")
    run_campaign_parser.add_argument("--first",type=int,default=0)
    run_campaign_parser.add_argument("--limit",type=int,default=8)
    run_campaign_parser.add_argument("--continue-on-error",action="store_true")
    run_campaign_parser.add_argument("--require-csim-pass",action="store_true",
        help="Opt-in: require passing source-attested CSim before Vivado")
    run_campaign_parser.add_argument("--vitis-executable",default="vitis_hls")
    run_campaign_parser.add_argument("--vivado-executable",default="vivado")
    audit=sub.add_parser("audit-campaign",help="Audit blocked cases and missing physical measurements")
    audit.add_argument("--directory",required=True)
    audit.add_argument("--required-evidence",choices=("E1","E2","E3","E4"),default="E1")
    audit.add_argument("--output")
    audit.add_argument("--fail-if-incomplete",action="store_true")
    prepare=sub.add_parser("prepare",help="Create source, testbench, Tcl, evidence manifests")
    prepare.add_argument("--onnx",required=True)
    prepare.add_argument("--input",required=True,help="NumPy .npy input tensor or sample array")
    prepare.add_argument("--labels",help="Optional integer class labels .npy")
    prepare.add_argument("--architecture")
    prepare.add_argument("--archive-inputs",action="store_true",help="Copy private ONNX/data inputs into the prepared case")
    prepare.add_argument("--export-ir",action="store_true",help="Save reconstructible Functional, Architectural and Lowered IR artifacts")
    prepare.add_argument("--case",required=True)
    prepare.add_argument("--part",default="xck26-sfvc784-2LV-c")
    prepare.add_argument("--clock-ns",type=float,default=5.0)
    prepare.add_argument("--tolerance",type=float,default=.1)
    for name,tool,timeout in (("run-vitis","vitis_hls",7200),("run-vivado","vivado",14400)):
        command=sub.add_parser(name,help=f"Opt-in local {tool} batch invocation")
        command.add_argument("--case",required=True)
        command.add_argument("--executable",default=tool)
        command.add_argument("--timeout-seconds",type=int,default=timeout)
    pareto=sub.add_parser("pareto-campaign",help="Calculate evidence-conditioned Pareto frontier; no vendor execution")
    pareto.add_argument("--directory",required=True)
    pareto.add_argument("--output",required=True)
    pareto.add_argument("--objectives",default="hls_latency_worst,postroute_lut,postroute_dsp")
    collect=sub.add_parser("collect",help="Validate source hashes and report evidence")
    collect.add_argument("--case",action="append",required=True)
    collect.add_argument("--output",required=True)
    campaign=sub.add_parser("prepare-campaign",help="Generate independent paper architecture cases; no vendor execution")
    campaign.add_argument("--onnx",required=True)
    campaign.add_argument("--input",required=True)
    campaign.add_argument("--plan",required=True,help="JSON: operator, default, variants, interactions")
    campaign.add_argument("--directory",required=True)
    campaign.add_argument("--labels",help="Optional integer class labels .npy")
    campaign.add_argument("--metadata",help="Optional training provenance file")
    campaign.add_argument("--archive-inputs",action="store_true",help="Copy private ONNX/data inputs into the campaign")
    collcampaign=sub.add_parser("collect-campaign",help="Collect attested measurements from prepared campaign cases")
    collcampaign.add_argument("--directory",required=True)
    collcampaign.add_argument("--output",required=True)
    collcampaign.add_argument("--tables-dir",help="Write CSV and Markdown paper tables")
    copy=sub.add_parser("archive-inputs",help="Opt-in private model/data capsule for a prepared experiment")
    copy.add_argument("--directory",required=True)
    copy.add_argument("--onnx",required=True)
    copy.add_argument("--input",required=True)
    copy.add_argument("--labels")
    copy.add_argument("--architecture")
    copy.add_argument("--plan")
    copy.add_argument("--metadata")
    check_archive=sub.add_parser("verify-inputs",help="Verify archived model/data bytes and optional preparation provenance")
    check_archive.add_argument("--directory",required=True)
    check_archive.add_argument("--prepared",action="store_true")
    full=sub.add_parser("prepare-full",help="Prepare all three Model IR stages, E0 oracle, E1 HLS and GCC checks")
    full.add_argument("--onnx",required=True)
    full.add_argument("--input",required=True)
    full.add_argument("--plan",required=True)
    full.add_argument("--directory",required=True)
    full.add_argument("--labels")
    full.add_argument("--host-samples",type=int,default=1)
    full.add_argument("--archive-inputs",action="store_true")
    full.add_argument("--allow-blocked",action="store_true")
    full.add_argument("--crosscheck-onnxruntime",action="store_true")
    full.add_argument("--trace-intermediates",action="store_true",help="Execute GCC trace for every operator output")
    auditfull=sub.add_parser("audit-full",help="Verify saved three-stage IR/HLS experiment receipts")
    auditfull.add_argument("--directory",required=True)
    auditfull.add_argument("--output")
    auditfull.add_argument("--require-host",action="store_true")
    auditfull.add_argument("--require-all-axes",action="store_true")
    auditfull.add_argument("--fail-if-incomplete",action="store_true")
    sweep=sub.add_parser("spec-precision-split",help="Prepare immutable fixed-point integer split study inputs; no vendor tools")
    sweep.add_argument("--suite",required=True,help="Original suite definition")
    sweep.add_argument("--prepared",required=True,help="Original prepared source study directory")
    sweep.add_argument("--directory",required=True,help="New destination for pinned assets and plans")
    sweep.add_argument("--width",type=int,default=8)
    sweep.add_argument("--integer-bits",type=int,action="append",default=None)
    suite=sub.add_parser("prepare-suite",help="Prepare multiple trained models as independent three-stage IR campaigns")
    suite.add_argument("--suite",required=True)
    suite.add_argument("--directory",required=True)
    suite.add_argument("--host-samples",type=int,default=1)
    suite.add_argument("--archive-inputs",action="store_true")
    suite.add_argument("--allow-missing",action="store_true")
    suite.add_argument("--allow-unsupported",action="store_true")
    suite.add_argument("--allow-blocked",action="store_true")
    suite.add_argument("--crosscheck-onnxruntime",action="store_true")
    suite.add_argument("--trace-intermediates",action="store_true",help="Check all operator tensors with GCC host floats")
    suite_audit=sub.add_parser("audit-suite",help="Recheck all model IR stages and native evidence in a suite")
    suite_audit.add_argument("--directory",required=True)
    suite_audit.add_argument("--require-host",action="store_true")
    suite_audit.add_argument("--require-all-axes",action="store_true")
    suite_audit.add_argument("--required-evidence",choices=("E1","E4"),default="E1",
        help="E1: source readiness; E4: actual CSim, HLS, routed timing for every architecture case")
    suite_audit.add_argument("--fail-if-incomplete",action="store_true")
    suite_audit.add_argument("--output")
    batch_audit=sub.add_parser("audit-suite-batch",
        help="Independently verify only a bounded selected E4/OOC/CSim batch, without vendor execution")
    batch_audit.add_argument("--directory",required=True)
    batch_audit.add_argument("--first",type=int,default=0)
    batch_audit.add_argument("--limit",type=int,default=2)
    batch_audit.add_argument("--required-evidence",choices=("E3","E4"),default="E4",
        help="E3: attested HLS plus passing CSim; E4: also route, timing and DRC")
    batch_audit.add_argument("--fail-if-incomplete",action="store_true")
    batch_audit.add_argument("--output",help="Optional fresh, immutable JSON audit receipt")
    fullrun=sub.add_parser("run-suite",help="Bounded dry-run or explicit native AMD execution across a prepared research suite")
    fullrun.add_argument("--directory",required=True)
    fullrun.add_argument("--stage",action="append",choices=("vitis","vivado"),required=True)
    fullrun.add_argument("--execute",action="store_true")
    fullrun.add_argument("--first",type=int,default=0)
    fullrun.add_argument("--limit",type=int,default=8)
    fullrun.add_argument("--continue-on-error",action="store_true")
    fullrun.add_argument("--require-csim-pass",action="store_true",
        help="Opt-in: require passing source-attested CSim before Vivado")
    fullrun.add_argument("--vitis-executable",default="vitis_hls")
    fullrun.add_argument("--vivado-executable",default="vivado")
    report_format=sub.add_parser("survey-hls-reports",
        help="Explore Vitis HLS text report layout without upgrading legacy evidence or executing tools")
    report_format.add_argument("--directory",required=True,
        help="Original pinned prepared research suite")
    report_format.add_argument("--inventory",required=True,
        help="Existing native_schedule_inventory.json from explain-ir-suite")
    report_format.add_argument("--output",required=True,
        help="Fresh immutable exploratory survey directory")
    redacted=sub.add_parser("probe-hls-table-grammar",
        help="Inspect only sanitized report header/row types; no vendor execution or performance values")
    redacted.add_argument("--directory",required=True)
    redacted.add_argument("--inventory",required=True,
        help="Pinned native_schedule_inventory.json")
    redacted.add_argument("--output",required=True,
        help="New immutable redacted HLS grammar output directory")
    redacted.add_argument("--model",action="append",default=None)
    redacted.add_argument("--case",action="append",default=None)
    classify=sub.add_parser("classify-hls-loop-tables",
        help="Read-only Vitis 2023.2 loop-table shape diagnostics; NO numerical II or evidence promotion")
    classify.add_argument("--directory",required=True)
    classify.add_argument("--inventory",required=True)
    classify.add_argument("--output",required=True,
        help="Fresh immutable redacted diagnostic output directory")
    classify.add_argument("--model",action="append",default=None)
    classify.add_argument("--case",action="append",default=None)
    triage=sub.add_parser("triage-hls-loop-gaps",
        help="Join source-attested IR research questions and redacted loop-table gaps without claiming achieved II")
    triage.add_argument("--schedule",required=True,help="ir_hls_schedule.json from explain-ir-suite")
    triage.add_argument("--inventory",required=True,help="native_schedule_inventory.json from same suite")
    triage.add_argument("--diagnostics",required=True,help="loop_table_diagnostics.json from classifier")
    triage.add_argument("--output",required=True,help="Fresh immutable weight-free gap report directory")
    replay=sub.add_parser("plan-native-hls-reruns",
        help="Read-only exact READY-index Vitis rerun plan from pinned native evidence")
    replay.add_argument("--directory",required=True,help="Original prepared suite directory")
    replay.add_argument("--inventory",required=True,help="Native report inventory for same suite")
    replay.add_argument("--triage",required=True,help="IR loop attribution-gap triage JSON")
    replay.add_argument("--output",required=True,help="Fresh immutable rerun-plan export directory")
    stage_replay=sub.add_parser("stage-native-hls-replay",
        help="Copy only source-verified E1 inputs to a fresh private Vitis campaign; no native run")
    stage_replay.add_argument("--directory",required=True,help="Original pinned prepared suite")
    stage_replay.add_argument("--inventory",required=True,help="Native report inventory from original suite")
    stage_replay.add_argument("--triage",required=True,help="Immutable attribution-gap report")
    stage_replay.add_argument("--output",required=True,help="Fresh private replay workspace root")
    audit_replay=sub.add_parser("audit-native-hls-replay",
        help="Read-only attestation audit of isolated HLS replay campaign")
    audit_replay.add_argument("--directory",required=True)
    audit_replay.add_argument("--output",required=True,help="Fresh immutable replay audit export")
    e3_compare=sub.add_parser("compare-isolated-hls-e3",
        help="Weight-free E3 HLS estimate ablations; never routed E4 or operator loop II")
    e3_compare.add_argument("--directory",required=True,
        help="Private source-identical isolated HLS replay")
    e3_compare.add_argument("--output",required=True,
        help="New immutable, weight-free E3 comparison directory")
    e3_interpret=sub.add_parser("interpret-isolated-hls-e3",
        help="Analyze equal HLS estimates, source digest identity and estimate Pareto tradeoffs")
    e3_interpret.add_argument("--comparison",required=True,
        help="Previously exported weight-free isolated_hls_e3_ablation.json")
    e3_interpret.add_argument("--output",required=True,
        help="New immutable weight-free interpretation output")
    tied_e3=sub.add_parser("diagnose-equal-hls-e3-sources",
        help="Source/manifest diagnostics for equal E3 estimates without private code export")
    tied_e3.add_argument("--directory",required=True,
        help="Private source-identical isolated HLS replay")
    tied_e3.add_argument("--comparison",required=True,
        help="Immutable weight-free isolated_hls_e3_ablation.json")
    tied_e3.add_argument("--output",required=True,
        help="Fresh immutable, privacy-redacted report directory")
    tied_rtl=sub.add_parser("inspect-tied-e3-rtl",
        help="Check fresh Vitis generated Verilog byte-set evidence for E3-tied pairs")
    tied_rtl.add_argument("--directory",required=True,
        help="Private source-attested isolated HLS replay")
    tied_rtl.add_argument("--comparison",required=True,
        help="Previously exported weight-free isolated_hls_e3_ablation.json")
    tied_rtl.add_argument("--output",required=True,
        help="Fresh immutable, weight-free RTL evidence export")
    paper_e3=sub.add_parser("summarize-paper-e3-evidence",
        help="Source-reaudited E1/E3/Verilog paper tables, without native execution")
    paper_e3.add_argument("--directory",required=True,
        help="Private isolated HLS replay with source-bound Vitis receipts")
    paper_e3.add_argument("--comparison",required=True,
        help="Unmodified source-attested isolated_hls_e3_ablation.json")
    paper_e3.add_argument("--output",required=True,
        help="Fresh immutable, weight-free publication evidence directory")
    quotient=sub.add_parser("analyze-e3-design-space-quotient",
        help="Source-reaudited distinct E1 C++/Verilog/E3 design classes, no native tools")
    quotient.add_argument("--directory",required=True,
        help="Private isolated HLS replay with source-attested Vitis RTL")
    quotient.add_argument("--comparison",required=True,
        help="Saved source-qualified E3 ablation JSON")
    quotient.add_argument("--output",required=True,
        help="New immutable, privacy-redacted design-space quotient export")
    matched_e4=sub.add_parser("prepare-matched-e4",
        help="Read-only source-attested E3 to E4 plan; optional canonical Vivado Tcl only")
    matched_e4.add_argument("--directory",required=True)
    matched_e4.add_argument("--comparison",required=True)
    matched_e4.add_argument("--output",required=True)
    matched_e4.add_argument("--stage-directory",default=None,
        help="Explicitly create a NEW source-only E1 physical workspace; never run tools")
    audit_e4=sub.add_parser("audit-matched-e4",
        help="Read-only eight-case E4 source/route/timing/DRC qualification, no vendor execution")
    audit_e4.add_argument("--directory",required=True)
    audit_e4.add_argument("--output",required=True,
        help="Fresh immutable source-attested physical audit directory")
    resume_e4=sub.add_parser("resume-matched-e4",
        help="Source-audited resumable physical campaign; read-only unless --execute")
    resume_e4.add_argument("--directory",required=True,
        help="NEW private, source-only E4 workspace; never publication E3 replay")
    resume_e4.add_argument("--execute",action="store_true",
        help="Explicitly execute pending Vitis then Vivado; skip previous receipts")
    resume_e4.add_argument("--max-cases",type=int,default=8,
        help="Bound new native cases per invocation (1..8; default 8)")
    cross_stage=sub.add_parser("analyze-cross-stage-e3-e4",
        help="Source-reaudited C++/RTL/E3/E4 observational pairs, no vendor execution")
    cross_stage.add_argument("--e3-directory",required=True,
        help="Private original immutable eight-case E3 replay")
    cross_stage.add_argument("--e4-directory",required=True,
        help="Private independently qualified eight-case routed E4 replay")
    cross_stage.add_argument("--output",required=True,
        help="Fresh privacy-safe immutable cross-stage paper export directory")
    ir_science=sub.add_parser("explain-ir-suite",
        help="Export weight-free, hash-verified Functional/Architectural/Lowered IR changes and E1/E4 outcomes")
    ir_science.add_argument("--directory",required=True,help="Prepared full research suite directory")
    ir_science.add_argument("--output",required=True,help="Fresh, immutable IR evidence export directory")
    suite_tables=sub.add_parser("collect-suite",help="Export cross-model scientific IR and measured native paper tables")
    suite_tables.add_argument("--directory",required=True)
    suite_tables.add_argument("--output",required=True)
    suite_tables.add_argument("--require-host",action="store_true")
    compare=sub.add_parser("compare-suite",help="Report attested within-model E4 ablation deltas from a collected suite snapshot")
    compare.add_argument("--tables",required=True,help="Existing collect-suite output directory")
    compare.add_argument("--output",required=True,help="Fresh comparison directory; never overwrites")
    numeric_compare=sub.add_parser("compare-precision-suites",help="Compare independently routed 8-bit study with same-model 12/16-bit baseline")
    numeric_compare.add_argument("--baseline-suite",required=True)
    numeric_compare.add_argument("--precision-suite",required=True)
    numeric_compare.add_argument("--model",required=True)
    numeric_compare.add_argument("--integer-bits",type=int,default=4)
    numeric_compare.add_argument("--output",required=True,help="New immutable comparison directory")
    mnist=sub.add_parser("assemble-mnist",help="Build private two-model MNIST research suite from newly trained exports")
    mnist.add_argument("--mlp-dir",required=True)
    mnist.add_argument("--cnn-dir",required=True)
    mnist.add_argument("--directory",required=True)
    mnist.add_argument("--skip-onnx-check",action="store_true")
    args=ap.parse_args(argv)
    if args.command=="assemble-mnist":
        from fpgai_research.experiments.mnist_assets import assemble_mnist_suite
        output=assemble_mnist_suite(args.mlp_dir,args.cnn_dir,args.directory,
            check_onnx=not args.skip_onnx_check)
        print(json.dumps(output,sort_keys=True,indent=2))
        return 0
    if args.command=="compare-precision-suites":
        from fpgai_research.experiments.cross_suite_precision import export_precision_comparison
        report=export_precision_comparison(
            baseline_suite=args.baseline_suite,precision_suite=args.precision_suite,
            model=args.model,integer_bits=args.integer_bits,output=args.output)
        print(json.dumps({k:v for k,v in report.items() if k!="cases"},
                         indent=2,sort_keys=True))
        return 0
    if args.command=="compare-suite":
        from fpgai_research.experiments.ablation_comparison import export_suite_comparisons
        report=export_suite_comparisons(args.tables,args.output)
        print(json.dumps({key:value for key,value in report.items() if key!="rows"},
                         indent=2,sort_keys=True))
        return 0
    if args.command=="survey-hls-reports":
        from fpgai_research.experiments.hls_report_format_survey import export_schedule_report_formats
        report=export_schedule_report_formats(
            args.directory,args.inventory,args.output)
        print(json.dumps({key:report[key] for key in (
            "schema","case_count","surveyed_report_count",
            "status_counts","distinct_format_count",
            "native_tools_executed","per_loop_schedule_evidence_available"
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="probe-hls-table-grammar":
        from fpgai_research.experiments.hls_report_schema_probe import export_hls_report_grammar
        report=export_hls_report_grammar(
            args.directory,args.inventory,args.output,
            selected_models=tuple(args.model or ()),
            selected_cases=tuple(args.case or ()))
        print(json.dumps({key:report[key] for key in (
            "schema","case_count","probed_report_count",
            "distinct_redacted_grammars","native_tools_executed",
            "per_loop_schedule_evidence_available"
        )},sort_keys=True,indent=2))
        return 0
    if args.command=="classify-hls-loop-tables":
        from fpgai_research.experiments.hls_loop_table_diagnostics import (
            export_vitis_loop_table_diagnostics,
        )
        report=export_vitis_loop_table_diagnostics(
            args.directory,args.inventory,args.output,
            selected_models=tuple(args.model or ()),
            selected_cases=tuple(args.case or ()))
        print(json.dumps({key:report[key] for key in (
            "schema","case_count","classified_report_count",
            "status_counts","native_tools_executed",
            "per_loop_schedule_evidence_available"
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="triage-hls-loop-gaps":
        from fpgai_research.experiments.loop_attribution_gaps import (
            export_loop_attribution_gaps,
        )
        result=export_loop_attribution_gaps(
            args.schedule,args.inventory,args.diagnostics,args.output)
        print(json.dumps({k:result[k] for k in (
            "schema","verified_question_count","diagnostic_case_count",
            "evidence_step_counts","native_tools_executed",
            "operator_schedule_attribution_verified"
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="plan-native-hls-reruns":
        from fpgai_research.experiments.native_hls_rerun_plan import export_hls_rerun_plan
        report=export_hls_rerun_plan(
            args.directory,args.inventory,args.triage,args.output)
        print(json.dumps({key:report[key] for key in (
            "schema","comparison_gap_count","distinct_priority_case_count",
            "priority_model_count","dry_run_command_count",
            "tools_executed","execution_authorized"
        )},sort_keys=True,indent=2))
        return 0
    if args.command=="stage-native-hls-replay":
        from fpgai_research.experiments.isolated_hls_replay import stage_isolated_hls_replay
        result=stage_isolated_hls_replay(
            args.directory,args.inventory,args.triage,args.output)
        print(json.dumps({key:result[key] for key in (
            "schema","selected_case_count","tools_executed",
            "inherited_native_receipts","private_inputs"
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="audit-native-hls-replay":
        from fpgai_research.experiments.isolated_hls_replay import export_isolated_hls_replay_audit
        result=export_isolated_hls_replay_audit(args.directory,args.output)
        print(json.dumps({key:result[key] for key in (
            "schema","case_count","e3_source_attested_case_count",
            "csim_passed_case_count","fresh_raw_hls_report_count",
            "fresh_loop_table_scanned_case_count",
            "fresh_alternative_grammar_probed_case_count",
            "fresh_loop_table_diagnostic_status_counts",
            "fully_e3_and_csim_qualified_count","native_tools_executed"
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="compare-isolated-hls-e3":
        from fpgai_research.experiments.isolated_hls_e3_ablation import (
            export_isolated_hls_e3_ablation,
        )
        report=export_isolated_hls_e3_ablation(args.directory,args.output)
        print(json.dumps({k:report[k] for k in (
            "schema","case_count","variant_count",
            "qualified_e3_hls_estimate_case_count","blocked_case_count",
            "native_tools_executed","operator_schedule_attribution_verified",
            "e4_physical_comparison_performed"
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="interpret-isolated-hls-e3":
        from fpgai_research.experiments.isolated_hls_e3_interpretation import (
            export_isolated_hls_e3_interpretation,
        )
        report=export_isolated_hls_e3_interpretation(
            args.comparison,args.output)
        print(json.dumps({k:report[k] for k in (
            "schema","case_count","dominated_hls_estimate_case_count",
            "equal_hls_estimate_pair_count","equal_cpp_source_pair_count",
            "source_identical_to_baseline_variant_count",
            "incomplete_hls_estimate_vector_count",
            "native_tools_executed","operator_schedule_attribution_verified",
            "e4_physical_comparison_performed",
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="diagnose-equal-hls-e3-sources":
        from fpgai_research.experiments.isolated_hls_e3_source_diagnostics import (
            export_equal_e3_source_diagnostics,
        )
        report=export_equal_e3_source_diagnostics(
            args.directory,args.comparison,args.output)
        print(json.dumps({k:report[k] for k in (
            "schema","equal_e3_estimate_pair_count",
            "byte_identical_cpp_pair_count",
            "nonweight_noncomment_text_equal_pair_count",
            "private_source_exported","weight_values_exported",
            "operator_schedule_attribution_verified",
            "native_tools_executed",
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="inspect-tied-e3-rtl":
        from fpgai_research.experiments.isolated_hls_e3_rtl_inventory import (
            export_tied_e3_rtl_inventory,
        )
        report=export_tied_e3_rtl_inventory(
            args.directory,args.comparison,args.output)
        print(json.dumps({k:report[k] for k in (
            "schema","equal_e3_estimate_pair_count",
            "fresh_comparable_pair_count","identical_rtl_file_set_pair_count",
            "different_rtl_file_set_pair_count","inconclusive_rtl_pair_count",
            "private_rtl_exported","operator_schedule_attribution_verified",
            "native_tools_executed",
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="summarize-paper-e3-evidence":
        from fpgai_research.experiments.isolated_e3_paper_evidence import (
            export_paper_e3_evidence,
        )
        report=export_paper_e3_evidence(
            args.directory,args.comparison,args.output)
        print(json.dumps({k:report[k] for k in (
            "schema","case_count","qualified_e3_case_count",
            "equal_hls_estimate_pair_count","dominated_hls_estimate_case_count",
            "rtl_evidence_class_counts","native_tools_executed",
            "operator_schedule_attribution_verified",
            "routed_e4_evidence_verified","weight_values_exported",
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="analyze-e3-design-space-quotient":
        from fpgai_research.experiments.isolated_e3_design_quotient import (
            export_e3_design_space_quotient,
        )
        outcome=export_e3_design_space_quotient(
            args.directory,args.comparison,args.output)
        print(json.dumps({k:outcome[k] for k in (
            "schema","model_count","case_count","within_model_pair_count",
            "native_tools_executed","routed_e4_comparison_performed",
            "formal_rtl_equivalence_verified",
            "private_rtl_filenames_or_hashes_exported",
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="prepare-matched-e4":
        from fpgai_research.experiments.matched_e4_preparation import export_matched_e4_preparation
        result=export_matched_e4_preparation(
            args.directory,args.comparison,args.output,stage_directory=args.stage_directory)
        print(json.dumps({k:result[k] for k in (
            "schema","case_count","model_count","staged_tcl_count",
            "native_vendor_tools_executed","native_e4_qualified",
            "historical_e3_receipts_overwritten",
            "original_e3_workspace_untouched","clean_independent_e1_workspace_created",
        )},indent=2,sort_keys=True))
        return 0
    if args.command=="audit-matched-e4":
        from fpgai_research.experiments.independent_e4_audit import audit_independent_e4
        result=audit_independent_e4(args.directory,args.output)
        print(json.dumps({k:result[k] for k in (
            "schema","case_count","e3_qualified_case_count",
            "routed_e4_attested_case_count","e4_qualified_case_count",
            "qualified_matched_e4_pair_count","planned_e4_pair_count",
            "vendor_tools_executed_by_audit","private_paths_or_digests_exported",
        )},sort_keys=True,indent=2))
        return 0
    if args.command=="resume-matched-e4":
        from fpgai_research.experiments.independent_e4_resume import resume_matched_e4
        outcome=resume_matched_e4(args.directory,execute=args.execute,
                                   max_cases=args.max_cases)
        print(json.dumps(outcome,sort_keys=True,indent=2))
        return int(outcome.get("executed") and bool(outcome.get("stop_reason")))
    if args.command=="analyze-cross-stage-e3-e4":
        from fpgai_research.experiments.cross_stage_observations import (
            export_cross_stage_pairs,
        )
        result=export_cross_stage_pairs(
            args.e3_directory,args.e4_directory,args.output)
        print(json.dumps({k:result[k] for k in (
            "schema","case_count","preregistered_pair_count",
            "e3_tied_pair_count","e3_tie_with_e4_difference_count",
            "physical_qor_tie_pair_count",
            "native_vendor_tools_executed_by_analysis",
        )},sort_keys=True,indent=2))
        return 0
    if args.command=="explain-ir-suite":
        from fpgai_research.experiments.ir_science import export_ir_suite
        report=export_ir_suite(args.directory,args.output)
        print(json.dumps({
            "schema":report["schema"],"case_count":report["case_count"],
            "models":[r["model"] for r in report["models"]],
            "e4_routed_cases":report["e4_routed_cases"],
            "qualified_cases":report["qualified_cases"],
            "source_suite_sha256":report["source_suite_sha256"],
            "weight_values_exported":report["weight_values_exported"],
            "native_tools_executed":report["native_tools_executed"],
        },sort_keys=True,indent=2))
        return 0
    if args.command=="collect-suite":
        from fpgai_research.experiments.suite_tables import collect_suite_tables
        report=collect_suite_tables(args.directory,args.output,require_host=args.require_host)
        print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2,sort_keys=True))
        return 0
    if args.command=="run-suite":
        from fpgai_research.experiments.suite_execution import run_full_suite
        receipt=run_full_suite(args.directory,stages=tuple(args.stage),
            execute=args.execute,first=args.first,limit=args.limit,
            continue_on_error=args.continue_on_error,
            require_csim_pass=args.require_csim_pass,
            vitis_executable=args.vitis_executable,
            vivado_executable=args.vivado_executable)
        print(json.dumps(receipt,indent=2,sort_keys=True))
        return int(args.execute and any(
            status.get("status") in ("FAILED","BLOCKED")
            for model in receipt["models"] for case in model["execution"]["cases"]
            for status in case["stages"].values()))
    if args.command=="spec-precision-split":
        from fpgai_research.experiments.precision_split import prepare_precision_split_spec
        report=prepare_precision_split_spec(
            suite_path=args.suite,prepared_suite=args.prepared,
            directory=args.directory,width=args.width,
            integer_splits=(args.integer_bits if args.integer_bits is not None
                            else (3,4,5)))
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0
    if args.command=="prepare-suite":
        from fpgai_research.experiments.full_suite import prepare_suite
        output=prepare_suite(suite_path=args.suite,directory=args.directory,
            host_samples=args.host_samples,archive_inputs=args.archive_inputs,
            allow_missing=args.allow_missing,allow_unsupported=args.allow_unsupported,
            require_all_legal=not args.allow_blocked,
            crosscheck_onnxruntime=args.crosscheck_onnxruntime,
            trace_intermediates=args.trace_intermediates)
        print(json.dumps(output,indent=2,sort_keys=True))
        return int(not output["all_models_ready"])
    if args.command=="audit-suite-batch":
        from fpgai_research.experiments.suite_native_batch import audit_suite_batch
        receipt=audit_suite_batch(args.directory,first=args.first,limit=args.limit,
                                  required_evidence=args.required_evidence)
        if args.output:
            target=Path(args.output)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError("Do not overwrite a prior native batch audit")
            target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        print(json.dumps(receipt,indent=2,sort_keys=True))
        return int(args.fail_if_incomplete and not receipt["all_selected_qualified"])
    if args.command=="audit-suite":
        from fpgai_research.experiments.full_suite import audit_suite
        native=args.required_evidence=="E4"
        result=audit_suite(args.directory,require_host=args.require_host or native,
                           require_all_axes=args.require_all_axes or native)
        qualified=(result["all_native_paper_experiments_complete"] if native
                   else result["all_models_source_ready"])
        result={**result,"requested_evidence_gate":args.required_evidence,
                "requested_evidence_gate_passed":qualified}
        if args.output:
            target=Path(args.output)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError("Do not overwrite existing study audit")
            target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps(result,indent=2,sort_keys=True))
        return int(args.fail_if_incomplete and not qualified)
    if args.command=="prepare-full":
        from fpgai_research.experiments.full_pipeline import prepare_full_experiment
        result=prepare_full_experiment(model=load_onnx(args.onnx),
            model_path=args.onnx,inputs_path=args.input,
            labels_path=args.labels,plan=json.loads(Path(args.plan).read_text()),
            directory=args.directory,host_samples=args.host_samples,
            require_all_legal=not args.allow_blocked,
            archive_inputs=args.archive_inputs,
            crosscheck_onnxruntime=args.crosscheck_onnxruntime,
            trace_intermediates=args.trace_intermediates)
        print(json.dumps({k:v for k,v in result.items()
            if k not in ("cases","reference_evaluation")},indent=2))
        return 0
    if args.command=="audit-full":
        from fpgai_research.experiments.full_audit import audit_full_experiment
        result=audit_full_experiment(args.directory,
            require_host=args.require_host,require_all_axes=args.require_all_axes)
        if args.output:
            path=Path(args.output)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                raise FileExistsError("Cannot overwrite an existing scientific audit")
            path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
        return int(args.fail_if_incomplete and not result["prepared_for_native_experiments"])
    if args.command=="archive-inputs":
        from fpgai_research.experiments.input_archive import archive_inputs, verify_archive
        archive_inputs(args.directory,model=args.onnx,samples=args.input,
                       labels=args.labels,architecture=args.architecture,
                       plan=args.plan,metadata=args.metadata)
        verified=verify_archive(args.directory)
        print(json.dumps({"directory":args.directory,"roles":verified["roles"],
                          "archived_bytes":verified["archived_bytes"],
                          "integrity_verified":True},indent=2))
        return 0
    if args.command=="verify-inputs":
        from fpgai_research.experiments.input_archive import verify_archive, verify_case_provenance
        verified=verify_case_provenance(args.directory) if args.prepared else verify_archive(args.directory)
        print(json.dumps(verified,indent=2))
        return 0
    if args.command=="run-campaign":
        from fpgai_research.experiments.execution import run_campaign
        receipt=run_campaign(args.directory,stages=tuple(args.stage),execute=args.execute,
            first=args.first,limit=args.limit,continue_on_error=args.continue_on_error,
            require_csim_pass=args.require_csim_pass,
            vitis_executable=args.vitis_executable,vivado_executable=args.vivado_executable)
        print(json.dumps(receipt,indent=2))
        return int(args.execute and any(
            state.get("status") in ("FAILED","BLOCKED")
            for case in receipt["cases"] for state in case["stages"].values()))
    if args.command=="pareto-campaign":
        from fpgai_research.experiments.pareto_campaign import analyze_campaign
        result=analyze_campaign(args.directory,args.output,tuple(args.objectives.split(",")))
        print(json.dumps({"eligible":result["eligible_cases"],
                          "frontier":[x["case"] for x in result["frontier"]],
                          "excluded":len(result["excluded"]),"output":args.output},indent=2))
        return 0
    if args.command=="reference-eval":
        from fpgai_research.experiments.reference_evaluation import evaluate_reference
        row=evaluate_reference(model=load_onnx(args.onnx), model_path=args.onnx,
              inputs_path=args.input,labels_path=args.labels,output_dir=args.output,
              crosscheck_onnxruntime=args.compare_onnxruntime)
        print(json.dumps(row,indent=2))
        return 0
    if args.command=="audit-campaign":
        from fpgai_research.experiments.reproduction_audit import audit_campaign
        row=audit_campaign(args.directory,required_evidence=args.required_evidence)
        if args.output:
            Path(args.output).write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
        print(json.dumps(row,indent=2))
        return int(args.fail_if_incomplete and not row["campaign_complete_for_required_evidence"])
    if args.command=="inspect-model":
        from fpgai_research.architecture.resolver import iteration_shape, ArchitectureError
        model=load_onnx(args.onnx)
        ops=[]
        for op in model.operators:
            try:
                shape=list(iteration_shape(model,op))
                blocker=None
            except ArchitectureError as exc:
                shape=None
                blocker={"code":exc.code,"detail":str(exc)}
            ops.append({"name":op.name,"kind":op.kind,
                        "iteration_shape":shape,"blocker":blocker})
        print(json.dumps({"inputs":list(model.inputs),"outputs":list(model.outputs),
                          "operators":ops},indent=2))
        return 0
    if args.command=="prepare":
        import numpy as np
        onnx=Path(args.onnx)
        model=load_onnx(str(onnx))
        x=np.load(args.input,allow_pickle=False)
        spec=json.loads(Path(args.architecture).read_text()) if args.architecture else {}
        if set(spec)-{"default","target","operators","regions","loops"}:
            raise ValueError("Unknown architecture specification keys")
        limits=TargetLimits(**spec.get("target",{}))
        if "regions" in spec or "loops" in spec:
            arch=resolve_scoped(model,spec.get("default",{}),
                    regions=spec.get("regions",{}),
                    operators=spec.get("operators",{}),
                    loops=spec.get("loops",{}),limits=limits)
        else:
            arch=resolve(model,_request(spec.get("default",{})),
              {k:_request(v) for k,v in spec.get("operators",{}).items()},limits)
        lowered=lower(arch)
        emitted=emit_hls(lowered)
        with staged_experiment(args.case) as case:
            record=write_vitis_package(str(case),model,emitted,x,
                       part=args.part,clock_ns=args.clock_ns,tolerance=args.tolerance,
                       labels=np.load(args.labels,allow_pickle=False) if args.labels else None)
            write_vivado_package(case,part=args.part,clock_ns=args.clock_ns)
            provenance={
                "onnx_sha256":sha256(onnx.read_bytes()).hexdigest(),
                "inputs_npy_sha256":sha256(Path(args.input).read_bytes()).hexdigest(),
                "labels_npy_sha256":sha256(Path(args.labels).read_bytes()).hexdigest() if args.labels else None,
                "architecture_sha256":sha256(Path(args.architecture).read_bytes()).hexdigest()
                                      if args.architecture else None,
                "trained_weights":"UNVERIFIED",
                "evaluation_dataset":"UNSPECIFIED",
                "note":"Add source model and dataset documentation before any paper accuracy claims."}
            (case/"inputs_provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
            if args.export_ir:
                from fpgai_research.model_ir.artifacts import write_ir_artifact
                artifact=write_ir_artifact(case/"progressive_ir.json",lowered)
                if artifact["stages"]["hls"]["manifest"]["cpp_sha256"]!=record["cpp_sha256"]:
                    raise ValueError("Progressive Model IR snapshot disagrees with prepared HLS source")
            if args.archive_inputs:
                from fpgai_research.experiments.input_archive import archive_inputs, verify_case_provenance
                archive_inputs(case,model=args.onnx,samples=args.input,labels=args.labels,
                               architecture=args.architecture)
                verify_case_provenance(case)
        print(json.dumps({"case":str(args.case),"evidence":"E1","cpp_sha256":record["cpp_sha256"]},indent=2))
        return 0
    if args.command=="run-vitis":
        record=run_vitis(args.case,executable=args.executable,timeout_seconds=args.timeout_seconds)
        print(json.dumps(record,indent=2))
        return 0 if record["completed_successfully"] else 1
    if args.command=="run-vivado":
        record=run_vivado(args.case,executable=args.executable,timeout_seconds=args.timeout_seconds)
        print(json.dumps(record,indent=2))
        return 0 if record["completed_successfully"] else 1
    if args.command=="prepare-campaign":
        import numpy as np
        source=Path(args.onnx)
        samples=Path(args.input)
        plan_path=Path(args.plan)
        plan=json.loads(plan_path.read_text())
        permitted={"operator","default","variants","interactions","target","part","clock_ns","tolerance"}
        if set(plan)-permitted or not {"operator","variants"} <= set(plan):
            raise ValueError("Campaign plan needs operator/variants; unexpected keys are blocked")
        provenance={"onnx_sha256":sha256(source.read_bytes()).hexdigest(),
                    "inputs_npy_sha256":sha256(samples.read_bytes()).hexdigest(),
                    "labels_npy_sha256":sha256(Path(args.labels).read_bytes()).hexdigest() if args.labels else None,
                    "plan_sha256":sha256(plan_path.read_bytes()).hexdigest()}
        if args.metadata:
            metadata_path=Path(args.metadata)
            provenance["metadata_sha256"]=sha256(metadata_path.read_bytes()).hexdigest()
            provenance["metadata"]=json.loads(metadata_path.read_text())
        with staged_experiment(args.directory) as stage:
            rows=prepare_campaign(model=load_onnx(str(source)),
                runtime_inputs=np.load(samples,allow_pickle=False),
                runtime_labels=np.load(args.labels,allow_pickle=False) if args.labels else None,
                directory=stage,operator=plan["operator"],
                default=_request(plan.get("default",{})),variants=plan["variants"],
                interactions=plan.get("interactions",[]),
                limits=TargetLimits(**plan.get("target",{})),
                part=plan.get("part","xck26-sfvc784-2LV-c"),
                clock_ns=plan.get("clock_ns",5.0),tolerance=plan.get("tolerance",0.1),
                source_provenance=provenance)
            if args.archive_inputs:
                from fpgai_research.experiments.input_archive import archive_inputs, verify_case_provenance
                archive_inputs(stage,model=args.onnx,samples=args.input,
                               labels=args.labels,plan=args.plan,metadata=args.metadata)
                verify_case_provenance(stage)
        print(json.dumps({"ready":sum(x["status"]=="READY" for x in rows),
                          "blocked":sum(x["status"]=="BLOCKED" for x in rows),
                          "evidence":"E1-source-only","directory":args.directory},indent=2))
        return 0
    if args.command=="collect-campaign":
        folder=Path(args.directory)
        index=json.loads((folder/"campaign.json").read_text())
        cases=[str(folder/x["case"]) for x in index["cases"] if x["status"]=="READY"]
        rows=export_results(cases,args.output)
        if args.tables_dir:
            export_tables(folder/"campaign.json",args.output,args.tables_dir)
        print(json.dumps({"cases":len(rows),"blocked":len(index["cases"])-len(rows),
                          "evidence":[x["evidence"] for x in rows]},indent=2))
        return 0
    if args.command=="collect":
        rows=export_results(args.case,args.output)
        print(json.dumps({"output":args.output,"evidence":[r["evidence"] for r in rows]},indent=2))
        return 0
    raise AssertionError("Unrecognized workflow action")

if __name__=="__main__":
    sys.exit(main())
