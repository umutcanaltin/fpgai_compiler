from __future__ import annotations

import argparse
from pathlib import Path

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


def export_stablehlo(out: Path) -> Path:
    import jax
    import jax.numpy as jnp

    spec = jax.ShapeDtypeStruct((1, 4), jnp.float32)
    exported = jax.export.export(jax.jit(model))(spec)
    text = str(exported.mlir_module())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the JAX linear example to StableHLO for FPGAI")
    parser.add_argument("--out", type=Path, default=Path("build/examples/frontends/jax_linear.mlir"))
    args = parser.parse_args()
    try:
        path = export_stablehlo(args.out)
    except Exception as exc:
        raise SystemExit(
            "JAX export requires JAX with StableHLO export support. "
            "Install the FPGAI framework MLIR optional dependencies first."
        ) from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
