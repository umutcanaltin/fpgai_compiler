"""Read original hls4ml Softmax strategy, type aliases and bundled arithmetic.

The old native CSim pair is independently replay-attested before any
source inspection. E1 model-owned C++/headers and the exact bundled
nnet_activation.h must byte-match both CSim clones.

This is an observation, NOT an operator trace or a source-level repair.
No Vitis/Vivado execution, code modification, numerical E2 promotion,
or public disclosure of original source/weights.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_native_softmax_source import (
    inspect_native_softmax_source, _firmware_files, MODE_RE,
)

SCHEMA="fpgai-research/native-hls4ml-softmax-semantic-audit/v1"
METHODS=("stable","latency","legacy","argmax")
ALIASES=("exp_table_t","inv_table_t","inv_inp_t","accum_t",
         "table_t","inp_norm_t")
HEADER=Path("firmware/nnet_utils/nnet_activation.h")
MEMBER=re.compile(r"\b(?:static\s+)?const\s+(?:unsigned|int)\s+"
                  r"(\w+)\s*=\s*(\d+)\s*;")
FIXED_DEF=re.compile(
    r"\btypedef\s+(ap_u?fixed)\s*<\s*(\d+)\s*,\s*(\d+)([^>]*)>"
    r"\s+([A-Za-z_]\w*)\s*;")
CONFIG=re.compile(
    r"\bstruct\s+(softmax_config\w*)\s*:\s*"
    r"nnet::(?:activ_config|softmax_config)\s*\{([^{}]*)\}\s*;",
    re.S)
CALL=re.compile(
    r"\bnnet::softmax\s*<\s*([A-Za-z_]\w*)\s*,"
    r"\s*([A-Za-z_]\w*)\s*,\s*(softmax_config\w*)\s*>")
IMPL=re.compile(
    r"\bimplementation\s*=\s*(?:nnet::)?softmax_implementation::"
    r"(stable|latency|legacy|argmax)\b")
ALIASED=re.compile(
    r"\btypedef\s+([A-Za-z_]\w*)\s+"
    r"(exp_table_t|inv_table_t|inv_inp_t|accum_t|table_t|inp_norm_t)\s*;")


def _uncomment(s):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", s, flags=re.S)


def _bounded_header(path):
    p=Path(path)
    if (p.is_symlink() or not p.is_file()
            or not 0<p.stat().st_size<=4*1024*1024):
        raise ValueError("Missing safe original bundled Softmax implementation header")
    raw=p.read_bytes()
    return raw,raw.decode("utf-8")


def _resolve_types(firmware,block,input_type,output_type):
    known={}
    for source in firmware.values():
        for match in FIXED_DEF.finditer(_uncomment(source["content"])):
            kind,w,i,extra,alias=match.groups()
            width,integer=int(w),int(i)
            if not 1<=width<=128 or not 0<=integer<=width:
                raise ValueError("Malformed generated ap_fixed type alias")
            record={
                "signed":kind=="ap_fixed",
                "width_bits":width,"integer_bits":integer,
                "fraction_bits":width-integer,
                "explicit_modes":MODE_RE.findall(extra),
            }
            if alias in known and known[alias]!=record:
                raise ValueError("Conflicting generated type alias")
            known[alias]=record
    role={}
    for target in ALIASES:
        matches=[m.group(1) for m in ALIASED.finditer(block)
                 if m.group(2)==target]
        if len(matches)>1:
            raise ValueError("Conflicting Softmax config type alias")
        role[target]={
            "declared_in_config":len(matches)==1,
            "precision":known.get(matches[0]) if matches else known.get(target),
        }
    return {
        "config_type_roles":role,
        "call_input_precision":known.get(input_type),
        "call_output_precision":known.get(output_type),
        "input_and_output_precision_resolved":
            input_type in known and output_type in known,
    }


def _parse_config(firmware):
    parameter=firmware.get("parameters.h")
    if not parameter:
        raise ValueError("Original generated parameters.h is absent")
    contents=_uncomment(parameter["content"])
    generated_calls=[]
    for row in firmware.values():
        generated_calls.extend(CALL.findall(_uncomment(row["content"])))
    configs=CONFIG.findall(contents)
    if len(generated_calls)!=1 or len(configs)<1:
        raise ValueError("Expected one generated hls4ml Softmax call/config")
    input_type,output_type,used=generated_calls[0]
    chosen=[block for name,block in configs if name==used]
    if len(chosen)!=1:
        raise ValueError("Cannot bind Softmax call to exactly one generated config")
    block=chosen[0]
    implementations=IMPL.findall(block)
    if len(implementations)!=1:
        raise ValueError("Generated config must select exactly one Softmax strategy")
    strategy=implementations[0]
    sizes={}
    for name,value in MEMBER.findall(block):
        if name in ("n_in","n_slice","table_size","exp_table_size",
                    "inv_table_size","exp_scale","axis"):
            if name in sizes:
                raise ValueError("Duplicate Softmax configuration integer")
            sizes[name]=int(value)
    if not sizes:
        raise ValueError("Missing generated Softmax dimension/table settings")
    return {
        "selected_implementation":strategy,
        "selected_config_bindings_verified":True,
        "declared_dimensions_and_tables":sizes,
        **_resolve_types(firmware,block,input_type,output_type),
    }


def _implementation_features(text,selected):
    cleaned=_uncomment(text)
    required=(
        "softmax_latency","softmax_stable","softmax_legacy",
        "softmax_argmax","softmax_idx_from_real_val")
    features={name:bool(re.search(r"\b"+name+r"\s*\(",cleaned))
              for name in required}
    if not features["softmax_"+selected]:
        raise ValueError("Bundled library lacks configured Softmax implementation")
    return {
        "library_implementation_functions_present":features,
        "index_helper_mentions_msb_slicing":(
            "x.width - 1" in cleaned and "x.width - N" in cleaned),
        "reciprocal_table_lookup_present":(
            "invert_table[" in cleaned and "inv_exp_sum" in cleaned),
        "exponential_times_reciprocal_product_present":bool(
            re.search(r"exp_res\[\s*i\s*\]\s*\*\s*inv_exp_sum",cleaned)),
        "selected_implementation_definition_present":True,
        "library_behavior_proven_by_text_scan":False,
    }


def inspect_native_softmax_semantics(stage_root,global_default,native_run):
    attested=inspect_native_softmax_source(
        stage_root,global_default,native_run)
    if (attested.get("native_pair_log_source_reattested") is not True
            or attested.get("original_e1_and_native_clone_headers_identical")
                is not True):
        raise ValueError("Original native Softmax source re-attestation failed")
    root=Path(stage_root)
    variants={}
    for rf in (1,2):
        key=f"rf{rf}"
        e1=root/f"e1_{key}"
        clone=root/f"csim_{key}"/"project"
        source=_firmware_files(e1)
        actual=_firmware_files(clone)
        if source.keys()!=actual.keys() or any(
                source[k]["sha256"]!=actual[k]["sha256"] for k in source):
            raise ValueError("Original E1 model C++/headers changed in native clone")
        original_h,template=_bounded_header(e1/HEADER)
        cloned_h,_=_bounded_header(clone/HEADER)
        if original_h!=cloned_h:
            raise ValueError("Original bundled Softmax library changed in native clone")
        config=_parse_config(source)
        variants[key]={
            **config,
            "bundled_library_sha256":sha256(original_h).hexdigest(),
            "bundled_library_source_bound":True,
            "template_features":_implementation_features(
                template,config["selected_implementation"]),
        }
    return {
        "schema":SCHEMA,"native_pair_source_and_log_reattested":True,
        "source_replay_receipt_sha256":
            attested["native_pair_replay_receipt_sha256"],
        "variants":variants,
        "same_selected_implementation":
            variants["rf1"]["selected_implementation"]==
            variants["rf2"]["selected_implementation"],
        "same_bundled_implementation_sha256":
            variants["rf1"]["bundled_library_sha256"]==
            variants["rf2"]["bundled_library_sha256"],
        "vendor_tools_executed_by_analysis":False,
        "native_e2_qualified_by_analysis":False,
        "numerical_root_cause_proven":False,
        "limits":(
            "Exact generated config and source-matched native bundled "
            "library are identified, not dynamically traced. "
            "Template string matches do not prove which arithmetic step "
            "lost probability mass. Do not alter E2 acceptance, and "
            "do not claim hardware QoR or source-revision parity "
            "across unrelated stages."),
    }


def export_native_softmax_semantics(stage_root,global_default,native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Choose fresh immutable private Softmax report directory")
    report=inspect_native_softmax_semantics(
        stage_root,global_default,native_run)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".hls4ml-softmax-semantics-",dir=dest.parent) as temp:
        work=Path(temp)/"report"
        work.mkdir()
        (work/"private_semantics.json").write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=[
            "# Native source-bound hls4ml Softmax strategy and arithmetic","",
            "Source-matched original generated headers, CSim clones, "
            "bundled library, and native Vitis logs were checked. "
            "No new vendor tools executed and no numerical E2 upgrade.","",
            "| RF | Strategy | Library SHA-256 | Sizes/lookup (literal) | Input precision | Output precision |",
            "|---|---|---|---|---|---|",
        ]
        def precision(v):
            if v is None:
                return "not resolved"
            return (f"{'signed' if v['signed'] else 'unsigned'} "
                    f"W={v['width_bits']} I={v['integer_bits']} "
                    f"F={v['fraction_bits']}")
        for label,v in report["variants"].items():
            lines.append(
                f"| {label} | {v['selected_implementation']} | "
                f"{v['bundled_library_sha256']} | "
                f"{json.dumps(v['declared_dimensions_and_tables'],sort_keys=True)} | "
                f"{precision(v['call_input_precision'])} | "
                f"{precision(v['call_output_precision'])} |")
        for label,v in report["variants"].items():
            lines.extend(["",f"## {label} — effective generated Softmax types","",
                "| Role | Defined in config | Resolved precision |",
                "|---|---|---|"])
            for role,item in v["config_type_roles"].items():
                lines.append(
                    f"| {role} | {item['declared_in_config']} | "
                    f"{precision(item['precision'])} |")
            lines.extend(["","Bundled source feature scan (not proof of runtime arithmetic):",""])
            for name,flag in v["template_features"].items():
                lines.append(f"- {name}: {flag}")
        lines.extend(["",
            "These are original SOURCE declarations and template feature "
            "signatures. The first numerically divergent operator is "
            "not demonstrated. Only controlled native CSim traces or "
            "an independently source-qualified test can establish "
            "whether reciprocal/table approximation produces the "
            "observed raw probability mass deficit.",""])
        (work/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Immutable Softmax destination appeared")
        work.rename(dest)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root",required=True)
    p.add_argument("--global-default",required=True)
    p.add_argument("--native-run",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    record=export_native_softmax_semantics(
        a.stage_root,a.global_default,a.native_run,a.output)
    print(json.dumps({
        "schema":SCHEMA,
        "source_attested":record["native_pair_source_and_log_reattested"],
        "methods":{k:v["selected_implementation"]
                   for k,v in record["variants"].items()},
        "template_hashes_match":
            record["same_bundled_implementation_sha256"],
        "vendor_tools_executed_by_analysis":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
