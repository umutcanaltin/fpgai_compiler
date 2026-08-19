from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

WEIGHT = (
    (0.10, 0.20, 0.30),
    (0.40, 0.50, 0.60),
    (0.70, 0.80, 0.90),
    (1.00, 1.10, 1.20),
)
BIAS = (0.01, 0.02, 0.03)


def model(x):
    """Equivalent JAX source model used by the cross-framework FPGAI example."""
    import jax.numpy as jnp

    weight = jnp.asarray(WEIGHT, dtype=jnp.float32)
    bias = jnp.asarray(BIAS, dtype=jnp.float32)
    return jnp.matmul(x, weight) + bias


def _stablehlo_boundary_names(text: str) -> tuple[str, str]:
    arg = re.search(r"func\.func[^\n]*\((%[A-Za-z0-9_.$-]+)\s*:", text)
    ret = re.search(r"(?:func\.)?return\s+(%[A-Za-z0-9_.$-]+)", text)
    if arg is None or ret is None:
        raise RuntimeError("Could not identify the StableHLO function input/output SSA values for the reference bundle")
    return arg.group(1).lstrip("%"), ret.group(1).lstrip("%")


def export_reference_bundle(*, stablehlo_text: str, out: Path) -> Path:
    import jax.numpy as jnp

    input_name, output_name = _stablehlo_boundary_names(stablehlo_text)
    x = np.asarray([[0.25, -0.50, 0.75, 1.25]], dtype=np.float32)
    y = np.asarray(model(jnp.asarray(x)), dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "input.npy", x)
    np.save(out / "output.npy", y)
    manifest = {
        "schema": "fpgai.frontend-reference/v1",
        "source_framework": "jax",
        "inputs": {"jax_arg0": {"path": "input.npy", "fpgai_tensor": input_name}},
        "outputs": {"jax_result": {"path": "output.npy", "fpgai_tensor": output_name}},
        "intermediates": {},
    }
    path = out / "reference_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def export_stablehlo(out: Path, *, reference_dir: Path | None = None) -> tuple[Path, Path]:
    import jax
    import jax.numpy as jnp

    spec = jax.ShapeDtypeStruct((1, 4), jnp.float32)
    exported = jax.export.export(jax.jit(model))(spec)
    text = str(exported.mlir_module())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    bundle_dir = reference_dir or out.with_name(out.stem + ".fpgai-reference")
    return out, export_reference_bundle(stablehlo_text=text, out=bundle_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the JAX linear example to StableHLO for FPGAI")
    parser.add_argument("--out", type=Path, default=Path("build/examples/frontends/jax_linear.mlir"))
    parser.add_argument(
        "--reference-dir", type=Path, default=None,
        help="Optional reference-bundle directory. Defaults to <model>.fpgai-reference for automatic FPGAI discovery.",
    )
    args = parser.parse_args()
    try:
        path, reference = export_stablehlo(args.out, reference_dir=args.reference_dir)
    except Exception as exc:
        raise SystemExit(
            "JAX export requires JAX with StableHLO export support. "
            "Install the FPGAI framework MLIR optional dependencies first."
        ) from exc
    print(path)
    print(reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
