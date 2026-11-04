"""Reproducible Vitis HLS Tcl/testbench and actual synthesis-report ingestion.

Generation is E1 evidence. Synthesis report parsing never launches vendor tools.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET
import json
from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.validation.reference import evaluate

def _floats(values):
    return ", ".join(f"{float(v):.9g}" for v in values)

def make_testbench(model: FunctionalModel, runtime_input, precision_bits: int,
                   integer_bits: int, tolerance: float = 0.1, labels=None) -> str:
    """Generate C simulation with finite numerical metrics over 1..256 samples."""
    import numpy as np
    from math import isfinite
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("C simulation supports one input and one output")
    shape=model.tensor(model.inputs[0]).shape
    if isinstance(runtime_input,np.ndarray) and runtime_input.dtype!=np.float32:
        raise ValueError("NumPy experiment samples must be float32; implicit numeric casts alter provenance")
    x=np.asarray(runtime_input,dtype=np.float32)
    from fpgai_research.validation.samples import normalize_samples
    samples=normalize_samples(x,shape)
    expected=np.stack([evaluate(model,{model.inputs[0]:item})[model.outputs[0]]
                       for item in samples],axis=0)
    if not np.all(np.isfinite(expected)):
        raise ValueError("Nonfinite reference output")
    label_header = label_init = label_update = label_footer = label_report = ""
    if labels is not None:
        provided=np.asarray(labels)
        output_shape=model.tensor(model.outputs[0]).shape
        if provided.dtype.kind not in ("i","u") or provided.ndim!=1 or len(provided)!=len(samples):
            raise ValueError("One integer class label required per sample")
        if len(output_shape)!=2 or output_shape[0]!=1:
            raise ValueError("Labeled C simulation requires output shape (1, classes)")
        if np.any(provided<0) or np.any(provided>=output_shape[1]):
            raise ValueError("Class label outside output range")
        label_header=("  const int truth[SAMPLES] = {" +
                      ",".join(str(int(x)) for x in provided) + "};\n" +
                      "  int compiled_correct=0, reference_correct=0, class_disagreements=0;")
        label_init=("    int best_compiled_index=0, best_reference_index=0;\n"
                    "    double best_compiled=-INFINITY, best_reference=-INFINITY;")
        label_update=("      if(observed > best_compiled) { best_compiled=observed; best_compiled_index=j; }\n"
                      "      if(reference > best_reference) { best_reference=reference; best_reference_index=j; }")
        label_footer=("    compiled_correct += (best_compiled_index == truth[sample]);\n"
                      "    reference_correct += (best_reference_index == truth[sample]);\n"
                      "    class_disagreements += (best_compiled_index != best_reference_index);")
        label_report=('  std::printf("FPGAI_CLASSIFICATION samples=%d compiled_correct=%d '
                      'reference_correct=%d class_disagreements=%d\\n", SAMPLES, '
                      'compiled_correct, reference_correct, class_disagreements);')
    nin=int(samples[0].size)
    nout=int(expected[0].size)
    inputs=",\n    ".join("{"+_floats(v.ravel())+"}" for v in samples)
    targets=",\n    ".join("{"+_floats(v.ravel())+"}" for v in expected)
    return f'''// FPGAI research model-only fixed-point numerical C simulation
#include <ap_fixed.h>
#include <cmath>
#include <cstdio>
using data_t = ap_fixed<{precision_bits}, {integer_bits}, AP_RND, AP_SAT>;
extern "C" void fpgai_model(const data_t*,data_t*);
int main() {{
  constexpr int SAMPLES={len(samples)};
  constexpr int INPUTS={nin};
  constexpr int OUTPUTS={nout};
  const float samples[SAMPLES][INPUTS] = {{ {inputs} }};
  const double expected[SAMPLES][OUTPUTS] = {{ {targets} }};
{label_header}
  int failures=0;
  double squared=0.,maximum=0.,dot=0.,nr=0.,no=0.;
  for(int sample=0;sample<SAMPLES;++sample) {{
    data_t input[INPUTS];
    data_t output[OUTPUTS] = {{}};
{label_init}
    for(int i=0;i<INPUTS;++i) input[i]=samples[sample][i];
    fpgai_model(input,output);
    for(int j=0;j<OUTPUTS;++j) {{
      const double observed=static_cast<double>(output[j]);
      const double reference=expected[sample][j];
      const double delta=observed-reference;
{label_update}
      if(!std::isfinite(observed)||std::fabs(delta)>{tolerance:.12g}) ++failures;
      squared+=delta*delta;
      if(std::fabs(delta)>maximum) maximum=std::fabs(delta);
      dot+=observed*reference;
      nr+=reference*reference;
      no+=observed*observed;
    }}
{label_footer}
  }}
  const double rmse=std::sqrt(squared/(SAMPLES*OUTPUTS));
  const double cosine=(nr*no>0) ? dot/std::sqrt(nr*no) : (rmse==0. ? 1. : 0.);
  std::printf("FPGAI_METRICS samples=%d outputs=%d cosine=%.12g rmse=%.12g maxabs=%.12g failures=%d\\n",
              SAMPLES,OUTPUTS,cosine,rmse,maximum,failures);
{label_report}
  std::printf("total_failures=%d\\n",failures);
  return failures ? 1 : 0;
}}
'''


def make_vitis_tcl(part: str = "xck26-sfvc784-2LV-c", clock_ns: float = 5.0,
                   run_csim: bool = True, run_csynth: bool = True,
                   clock_uncertainty_ns: float | None = None) -> str:
    """Stable public writer API; implementation lives in pure toolchain Tcl."""
    from .tcl import make_vitis_tcl as generate
    return generate(
        part=part, clock_ns=clock_ns, run_csim=run_csim,
        run_csynth=run_csynth,
        clock_uncertainty_ns=clock_uncertainty_ns)


def write_vitis_package(output: str, model: FunctionalModel, emitted,
                        runtime_input, part: str = "xck26-sfvc784-2LV-c",
                        clock_ns: float = 5.0, tolerance: float = 0.1, labels=None,
                        clock_uncertainty_ns: float | None = None):
    from fpgai_research.model_ir.semantics import functional_fingerprint
    if emitted.manifest.get("functional_sha256") != functional_fingerprint(model):
        raise ValueError("HLS source and C-simulation reference describe different Functional Model IR weights/operators")
    if emitted.manifest.get("cpp_sha256") != sha256(emitted.cpp.encode("utf-8")).hexdigest():
        raise ValueError("Generated HLS source differs from the emission manifest")
    dest = Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty HLS case; use a fresh experiment directory")
    # Validate numerical input and target Tcl before publishing any source files.
    bench_source=make_testbench(model,runtime_input,emitted.manifest["precision_bits"],
                               emitted.manifest["integer_bits"],tolerance,labels=labels)
    tcl_source=make_vitis_tcl(part, clock_ns,
                             clock_uncertainty_ns=clock_uncertainty_ns)
    dest.mkdir(parents=True,exist_ok=True)
    (dest / "model.cpp").write_text(emitted.cpp)
    (dest / "tb.cpp").write_text(bench_source)
    (dest / "run_hls.tcl").write_text(tcl_source)
    from math import prod
    from fpgai_research.validation.samples import normalize_samples
    expected_count = len(normalize_samples(runtime_input, model.tensor(model.inputs[0]).shape))
    expected_outputs = prod(model.tensor(model.outputs[0]).shape)
    record = {**emitted.manifest, "part": part, "clock_ns": clock_ns,
              "csim_sample_count": expected_count, "csim_outputs_per_sample": expected_outputs,
              "testbench_sha256": sha256((dest / "tb.cpp").read_bytes()).hexdigest(),
              "numerical_tolerance": tolerance,
              "labeled_sample_count": len(labels) if labels is not None else None,
              "evidence": "E1-source-ready-for-vitis"}
    if clock_uncertainty_ns is not None:
        record["explicit_clock_uncertainty_ns"] = float(clock_uncertainty_ns)
    (dest / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record

def parse_csynth_xml(path: str):
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"No actual synthesis report: {file}")
    root = ET.parse(file).getroot()
    # Only use estimated *consumed* resources, never device AvailableResources.
    resources = root.find(".//AreaEstimates/Resources")
    latency = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    if resources is None or latency is None:
        raise ValueError("Vitis HLS report lacks area estimates or latency summary")

    def read(node, tag):
        child = node.find(tag)
        return child.text.strip() if child is not None and child.text else None

    fields = {"latency_best": read(latency, "Best-caseLatency"),
              "latency_worst": read(latency, "Worst-caseLatency"),
              "ii_min": read(latency, "Interval-min"),
              "ii_max": read(latency, "Interval-max"),
              "lut": read(resources, "LUT"), "ff": read(resources, "FF"),
              "dsp": read(resources, "DSP"),
              "bram_18k": read(resources, "BRAM_18K"),
              "uram": read(resources, "URAM")}
    # Do not use an incomplete XML stub as a paper-quality synthesis result.
    # Some HLS releases report '-' for unbounded latency; in that case an E3
    # timing/latency claim must be withheld until a bounded report is available.
    if fields["latency_best"] is None or all(fields[name] is None for name in ("lut", "ff", "dsp", "bram_18k", "uram")):
        raise ValueError("HLS report needs measured best-case latency and at least one used-resource field")
    numeric = {}
    for name, literal in fields.items():
        if literal is None:
            numeric[name] = None
            continue
        if not literal.isdecimal():
            raise ValueError(f"HLS {name} must be a nonnegative integer; found {literal!r}")
        numeric[name] = int(literal)
    if numeric["latency_worst"] is not None and numeric["latency_worst"] < numeric["latency_best"]:
        raise ValueError("HLS worst-case latency is below best-case latency")
    if numeric["ii_min"] is not None and numeric["ii_min"] < 1:
        raise ValueError("HLS achieved initiation interval must be positive")
    if numeric["ii_max"] is not None and (numeric["ii_max"] < 1 or
        (numeric["ii_min"] is not None and numeric["ii_max"] < numeric["ii_min"])):
        raise ValueError("HLS maximum initiation interval is inconsistent")
    # Preserve raw numeric strings for existing CSV consumers; validation
    # is independent of presentation/formatting policy.
    return {"evidence": "E3-HLS-synthesis-report", "report_sha256": sha256(file.read_bytes()).hexdigest(),
            "report_path": str(file), "measurements": fields}
