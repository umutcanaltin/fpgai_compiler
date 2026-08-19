from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export maintained JAX examples to textual StableHLO for FPGAI")
    parser.add_argument("--kind", choices=("attention", "attention_rmsnorm", "rmsnorm", "layernorm"), default="attention")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--head-dimension", type=int, default=8)
    args = parser.parse_args()

    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        raise SystemExit(
            "JAX is required for this exporter. Install the FPGAI framework MLIR frontend with: "
            "python -m pip install -e '.[framework-mlir]'"
        ) from exc

    s = int(args.sequence_length); d = int(args.head_dimension)
    if s <= 0 or d <= 0:
        raise SystemExit("sequence length and head dimension must be positive")

    if args.kind in {"attention", "attention_rmsnorm"}:
        if args.kind == "attention":
            def function(q, k, v):
                scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (1.0 / jnp.sqrt(float(d)))
                probabilities = jax.nn.softmax(scores, axis=-1)
                return jnp.matmul(probabilities, v)
            specs = [jax.ShapeDtypeStruct((1, s, d), jnp.float32)] * 3
        else:
            def function(q, k, v, scale):
                scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (1.0 / jnp.sqrt(float(d)))
                probabilities = jax.nn.softmax(scores, axis=-1)
                context = jnp.matmul(probabilities, v)
                return context * jax.lax.rsqrt(jnp.mean(context * context, axis=-1, keepdims=True) + 1e-5) * scale
            specs = [jax.ShapeDtypeStruct((1, s, d), jnp.float32)] * 3 + [jax.ShapeDtypeStruct((d,), jnp.float32)]
    elif args.kind == "rmsnorm":
        def function(x, scale):
            return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-5) * scale
        specs = [jax.ShapeDtypeStruct((1, s, d), jnp.float32), jax.ShapeDtypeStruct((d,), jnp.float32)]
    else:
        def function(x, scale, bias):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            centered = x - mean
            variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
            return centered * jax.lax.rsqrt(variance + 1e-5) * scale + bias
        specs = [
            jax.ShapeDtypeStruct((1, s, d), jnp.float32),
            jax.ShapeDtypeStruct((d,), jnp.float32),
            jax.ShapeDtypeStruct((d,), jnp.float32),
        ]

    exported = jax.export.export(jax.jit(function))(*specs)
    stablehlo = str(exported.mlir_module())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(stablehlo + ("" if stablehlo.endswith("\n") else "\n"), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
