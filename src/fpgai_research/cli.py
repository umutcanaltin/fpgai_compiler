"""Local ONNX -> HLS C++ compilation. Does not invoke synthesis automatically."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.architecture.resolver import resolve, TargetLimits
from fpgai_research.compiler import compile_architecture
from fpgai_research.architecture.scopes import resolve_scoped


def _request(obj: dict) -> ArchitectureRequest:
    keys = {f.name for f in fields(ArchitectureRequest)}
    if set(obj) - keys:
        raise ValueError(f"Unknown architecture keys: {sorted(set(obj) - keys)}")
    if "tile" in obj:
        obj = {**obj, "tile": tuple(obj["tile"])}
    return ArchitectureRequest(**obj)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compile one static ONNX model to Vitis HLS C++")
    ap.add_argument("model", help="Input ONNX model")
    ap.add_argument("-o", "--output", required=True, help="Output directory")
    ap.add_argument("--architecture", help="JSON object: default contract and operator overrides")
    ap.add_argument("--export-ir", action="store_true", help="Export full reconstructible progressive Model IR stages (includes model weights)")
    args = ap.parse_args(argv)
    contract = json.loads(Path(args.architecture).read_text()) if args.architecture else {}
    if set(contract) - {"default", "operators", "target", "regions", "loops"}:
        ap.error("Unknown top-level architecture contract key")
    limits = TargetLimits(**contract.get("target", {}))
    model = load_onnx(args.model)
    if "regions" in contract or "loops" in contract:
        architecture = resolve_scoped(model, contract.get("default", {}),
              regions=contract.get("regions", {}), operators=contract.get("operators", {}),
              loops=contract.get("loops", {}), limits=limits)
    else:
        architecture = resolve(model, _request(contract.get("default", {})),
                               {key: _request(val) for key, val in contract.get("operators", {}).items()}, limits)
    compiled = compile_architecture(architecture)
    lowered = compiled.lowered
    result = compiled.emission
    output = Path(args.output)
    if args.export_ir and output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a versioned Model IR experiment")
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.cpp").write_text(result.cpp)
    (output / "manifest.json").write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n")
    if args.export_ir:
        from fpgai_research.model_ir.artifacts import write_ir_artifact
        write_ir_artifact(output / "progressive_model_ir.json", lowered)
    print(f"Generated {output / 'model.cpp'} (E1; no synthesis or numerical validation claimed)")


if __name__ == "__main__":
    main()
