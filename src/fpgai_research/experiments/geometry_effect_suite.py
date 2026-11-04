"""Preregister small-to-larger MatMul/Conv seven-knob hardware sensitivity.

These are deterministic SYNTHETIC operator probes. They are NOT MNIST
networks, retrained models, or evidence of model accuracy. Source/model
and input bytes are frozen and hashed. No Vitis/Vivado work occurs here.
All main effects, 21 pairwise and three higher-order requests come from
the existing comprehensive seven-axis plan factory. Invalid points must
remain blocked by the normal source compiler, never silently rewritten.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_suite import (
    SCHEMA as SUITE_SCHEMA, validate_suite,
)
from fpgai_research.experiments.full_pipeline import parse_plan
from fpgai_research.experiments.plan_factory import make_comprehensive_seven_axis_plan
from fpgai_research.experiments.campaign import DIMENSIONS
from fpgai_research.frontend.onnx import load_onnx

SCHEMA = "fpgai-research/geometry-controlled-seven-axis-probes/v1"
PROFILES = ("pilot", "full")
DEFAULT_PART = "xck26-sfvc784-2LV-c"
# Bounded dimensions: MLP GEMM/MatMul (M,N,K) and NCHW Conv (Cin,Cout,H,W).
GEOMETRIES = (
    ("probe_matmul_08", "MatMul", "small", (1, 8, 8)),
    ("probe_conv_08", "Conv", "small", (2, 4, 8, 8)),
    ("probe_matmul_16", "MatMul", "medium", (1, 16, 16)),
    ("probe_conv_12", "Conv", "medium", (4, 8, 12, 12)),
    ("probe_matmul_32", "MatMul", "large", (1, 32, 32)),
    ("probe_conv_16", "Conv", "large", (4, 8, 16, 16)),
)


def _assets(kind, shape):
    """Return exactly-representable FP32 samples and bounded weights."""
    import numpy as np
    if kind=="MatMul":
        m,n,k=shape
        input_shape=(m,k)
        weight_shape=(k,n)
        output_shape=(m,n)
    else:
        cin,cout,h,w=shape
        input_shape=(1,cin,h,w)
        weight_shape=(cout,cin,3,3)
        output_shape=(1,cout,h-2,w-2)
    count=prod(input_shape)
    x=np.where(np.arange(count)%3==0,1.0,0.5).astype(np.float32).reshape(input_shape)
    nweights=prod(weight_shape)
    w=np.where(np.arange(nweights)%4==0,-0.25,0.25).astype(
        np.float32).reshape(weight_shape)
    return x,w,output_shape


def _write_onnx(path,kind,shape,operator):
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
    except ImportError as exc:
        raise RuntimeError("ONNX is required: pip install -e '.[onnx]'") from exc
    x,w,output_shape=_assets(kind,shape)
    graph=helper.make_graph(
        [helper.make_node(kind,["input","weight"],["output"],name=operator)],
        "synthetic_geometry_probe_"+operator,
        [helper.make_tensor_value_info("input",TensorProto.FLOAT,list(x.shape))],
        [helper.make_tensor_value_info("output",TensorProto.FLOAT,list(output_shape))],
        [numpy_helper.from_array(w,name="weight")],
    )
    model=helper.make_model(graph,opset_imports=[
        helper.make_operatorsetid("",13)])
    # Bound ONNX IR for compatibility with scientific ORT installations.
    model.ir_version=8
    onnx.checker.check_model(model)
    onnx.save(model,str(path))
    return x


def _verify_all_dimensions(plan):
    parsed=parse_plan(plan)
    seen={name for name,_ in parsed["points"]}
    if not set(DIMENSIONS).issubset(seen) or "baseline" not in seen:
        raise ValueError("Controlled probe omits one or more seven dimensions")
    from itertools import combinations
    expected={frozenset(pair) for pair in combinations(DIMENSIONS,2)}
    paired={frozenset(d) for d in plan["interactions"] if len(d)==2}
    if paired!=expected:
        raise ValueError("Controlled probe does not cover all 21 dimension pairs")
    return len(parsed["points"])


def output_quantization_floor(model, sample, *, precision_bits=8,
                              integer_bits=6, tolerance=0.05):
    """Lower bound on output error for the declared fixed-point grid.

    This is NOT fixed-point CSim or proof of numerical correctness: the
    actual circuit may have additional intermediate rounding or overflow.
    But an output that is more than tolerance from every representable
    fixed-point value cannot pass the existing max-error criterion.
    """
    import numpy as np
    from fpgai_research.validation.reference import evaluate

    if (type(precision_bits) is not int or type(integer_bits) is not int
            or not 1<=integer_bits<=precision_bits
            or type(tolerance) not in (int,float) or tolerance<=0):
        raise ValueError("Invalid quantization-grid parameters")
    reference=np.asarray(evaluate(model,{model.inputs[0]:sample})[
        model.outputs[0]],dtype=np.float64)
    if not np.all(np.isfinite(reference)) or not reference.size:
        raise ValueError("Finite reference output required")
    quantum=2.0**(integer_bits-precision_bits)
    minimum=-float(2**(integer_bits-1))
    maximum=float(2**(integer_bits-1))-quantum
    nearest=np.clip(np.rint(reference/quantum)*quantum,minimum,maximum)
    lower_bound=np.abs(nearest-reference)
    return {
        "precision_bits":precision_bits,
        "integer_bits":integer_bits,
        "output_quantization_step":quantum,
        "maximum_unavoidable_absolute_output_error":
            float(lower_bound.max()),
        "outputs_unavoidably_outside_tolerance":
            int(np.count_nonzero(lower_bound>tolerance)),
        "output_count":int(reference.size),
        "tolerance":float(tolerance),
        "numerical_acceptance_impossible_by_output_grid":
            bool(np.any(lower_bound>tolerance)),
        "only_a_necessary_output_representability_preflight":True,
        "source_attested_fixed_point_csim_passed":False,
    }


def assemble_geometry_effect_suite(output, *, profile="pilot",
                                   part=DEFAULT_PART, clock_ns=5.0,
                                   integer_bits=6, only_kind=None,
                                   only_size=None):
    """Make an immutable suite accepted by the existing native workflow."""
    if profile not in PROFILES:
        raise ValueError("Expected pilot or full geometry profile")
    if not isinstance(part,str) or not part:
        raise ValueError("Target FPGA part must be nonempty")
    if type(clock_ns) not in (int,float) or not 0<clock_ns<1000:
        raise ValueError("Require finite practical positive clock in ns")
    # All width variants include 8 bits, hence the signed integer field
    # must leave at least one fractional bit. This parameter is fixed
    # across the seven-axis study; varying it within a precision sweep
    # would confound the interpretation.
    if type(integer_bits) is not int or not 1<=integer_bits<=7:
        raise ValueError("integer_bits must be 1..7 for the 8-bit sweep")
    if only_kind is not None and only_kind not in ("MatMul","Conv"):
        raise ValueError("only_kind must be MatMul, Conv or omitted")
    if only_size is not None and only_size not in ("small","medium","large"):
        raise ValueError("only_size must be small, medium, large or omitted")
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Geometry experiment output must be a NEW directory")
    shapes=GEOMETRIES[:2] if profile=="pilot" else GEOMETRIES
    if only_kind is not None:
        shapes=tuple(x for x in shapes if x[1]==only_kind)
    if only_size is not None:
        shapes=tuple(x for x in shapes if x[2]==only_size)
    if not shapes:
        raise ValueError("No geometry selected")
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent,prefix=".geometry-probes-") as temp:
        stage=Path(temp)/"suite"
        (stage/"inputs").mkdir(parents=True)
        (stage/"plans").mkdir()
        entries=[]
        hashes={}
        studies=[]
        for ident,kind,size,geometry in shapes:
            operator="compute"
            rel_onnx="inputs/"+ident+".onnx"
            rel_input="inputs/"+ident+".npy"
            rel_plan="plans/"+ident+".json"
            model_path=stage/rel_onnx
            samples=_write_onnx(model_path,kind,geometry,operator)
            import numpy as np
            np.save(stage/rel_input,samples[np.newaxis,...])
            model=load_onnx(str(model_path))
            model.validate()
            plan=make_comprehensive_seven_axis_plan(
                model,operator=operator,part=part,clock_ns=clock_ns)
            plan["default"]["integer_bits"]=integer_bits
            if kind=="Conv":
                # The existing generic planner proposes a second tile that
                # changes Cin/Kh/Kw and is rejected by the current Conv
                # resolver. In this *new* probe only, constrain both tile
                # levels to dimensions the compiler actually supports:
                # output channels, output height and output width.
                first=plan["variants"]["tile"][0]
                second=[
                    first[0],
                    max(1,first[1]//2),
                    max(1,first[2]//2),
                    max(1,first[3]//2),
                    first[4],first[5],first[6],
                ]
                if second==first:
                    raise ValueError("Conv geometry too small for two distinct legal tile levels")
                plan["variants"]["tile"]=[first,second]
            points=_verify_all_dimensions(plan)
            grid=output_quantization_floor(
                model,samples,precision_bits=8,
                integer_bits=integer_bits,tolerance=plan["tolerance"])
            (stage/rel_plan).write_text(json.dumps(
                plan,indent=2,sort_keys=True,allow_nan=False)+"\n")
            entries.append({
                "id":ident,"family":"other",
                "onnx":rel_onnx,"input":rel_input,
                "plan":rel_plan,"operator":operator,
            })
            studies.append({
                "id":ident,"kind":kind,"size":size,"geometry":list(geometry),
                "preregistered_architecture_requests":points,
                "seven_main_axes_requested":list(DIMENSIONS),
                "all_21_pairs_requested":True,
                "eight_bit_output_quantization_floor":grid,
                "physical_effects_established":False,
            })
            for label,relative in (("onnx",rel_onnx),("input",rel_input),
                                   ("plan",rel_plan)):
                hashes[relative]=sha256((stage/relative).read_bytes()).hexdigest()
        suite={
            "schema":SUITE_SCHEMA,
            "notes":(
                "Deterministic SYNTHETIC size-controlled MatMul/Conv hardware "
                "probes. NOT original-trained MNIST, not accuracy experiments. "
                "No vendor execution; unsupported points remain blockers."),
            "experiments":entries,
        }
        validate_suite(suite)
        (stage/"suite.json").write_text(
            json.dumps(suite,indent=2,sort_keys=True)+"\n")
        receipt={
            "schema":SCHEMA,
            "profile":profile,
            "only_kind":only_kind,
            "only_size":only_size,
            "target_part":part,"clock_ns":clock_ns,
            "integer_bits_fixed_across_all_widths":integer_bits,
            "source_suite_sha256":sha256((stage/"suite.json").read_bytes()).hexdigest(),
            "source_asset_sha256":hashes,
            "studies":studies,
            "study_count":len(studies),
            "weights_are_deterministic_synthetic_not_trained":True,
            "original_mnist_models_or_weights_modified":False,
            "numerical_csim_executed":False,
            "vitis_or_vivado_executed":False,
            "routed_qor_qualified":False,
            "e1_ir_prepared":False,
            "scientific_note":(
                "This is a preregistration, not a result. Native effects "
                "require actual per-case compiler output and independent "
                "CSim/E3/E4 gates; sizes are a separate experimental factor. "
                "The 8-bit output grid preflight is a necessary lower bound "
                "only and cannot establish numerical acceptance."),
        }
        (stage/"source_receipt.json").write_text(
            json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Geometry destination appeared")
        stage.rename(output)
    return receipt


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--profile",choices=PROFILES,default="pilot")
    parser.add_argument("--part",default=DEFAULT_PART)
    parser.add_argument("--clock-ns",type=float,default=5.0)
    parser.add_argument("--integer-bits",type=int,default=6,
                        help="Fixed signed integer bits (1..7) at every precision width")
    parser.add_argument("--only-kind",choices=("MatMul","Conv"),
                        help="Preregister only one operator family (e.g. Conv)")
    parser.add_argument("--only-size",choices=("small","medium","large"),
                        help="Preregister only one operator geometry size")
    args=parser.parse_args(argv)
    receipt=assemble_geometry_effect_suite(
        args.output,profile=args.profile,part=args.part,
        clock_ns=args.clock_ns,integer_bits=args.integer_bits,
        only_kind=args.only_kind,only_size=args.only_size)
    print(json.dumps({"schema":receipt["schema"],
                      "profile":receipt["profile"],
                      "integer_bits_fixed_across_all_widths":
                          receipt["integer_bits_fixed_across_all_widths"],
                      "study_count":receipt["study_count"],
                      "requested_cases":sum(
                          x["preregistered_architecture_requests"]
                          for x in receipt["studies"]),
                      "vendor_tools_executed":False},
                     sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
