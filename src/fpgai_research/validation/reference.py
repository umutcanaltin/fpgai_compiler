"""Independent NumPy float32 model oracle; no fixed-point equivalence claim."""
from __future__ import annotations
from fpgai_research.model_ir.contracts import FunctionalModel

def evaluate(model: FunctionalModel, inputs: dict, *, return_all: bool = False):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install numpy for reference evaluation") from exc
    model.validate()
    tensors = {}
    for t in model.tensors:
        if t.data is not None:
            tensors[t.name] = np.asarray(t.data, dtype=np.float32).reshape(t.shape)
    for name in model.inputs:
        if name not in inputs:
            raise ValueError(f"Missing runtime tensor: {name}")
        value = np.asarray(inputs[name], dtype=np.float32)
        if tuple(value.shape) != model.tensor(name).shape:
            raise ValueError(f"Input {name}: shape mismatch")
        tensors[name] = value
    for op in model.operators:
        v = [tensors[key] for key in op.inputs]
        if op.kind in ("MatMul", "Gemm"):
            result = v[0] @ (v[1].T if op.kind == "Gemm" and op.attr("transB", 0) else v[1])
            if len(v) == 3:
                result = result + v[2]
        elif op.kind == "Relu":
            result = np.maximum(v[0], 0)
        elif op.kind == "LeakyRelu":
            result = np.where(v[0] >= 0, v[0], v[0] * np.float32(op.attr("alpha", 0.01)))
        elif op.kind == "Concat":
            result = np.concatenate(v, axis=1)
        elif op.kind == "BatchNormalization":
            eps = np.float32(op.attr("epsilon", 1e-5))
            scale, bias, mean, variance = (np.asarray(a, dtype=np.float32).reshape(1, -1, 1, 1) for a in v[1:])
            result = (v[0] - mean) / np.sqrt(variance + eps) * scale + bias
        elif op.kind in ("Identity", "Flatten", "Reshape", "Squeeze", "Unsqueeze"):
            result = v[0].reshape(model.tensor(op.outputs[0]).shape)
        elif op.kind == "Add":
            result = v[0] + v[1]
        elif op.kind == "Mul":
            result = v[0] * v[1]
        elif op.kind == "Sigmoid":
            result = 1.0 / (1.0 + np.exp(-v[0]))
        elif op.kind == "Transpose":
            result = v[0].T
        elif op.kind == "GlobalAveragePool":
            result = v[0].mean(axis=(2, 3), keepdims=True)
        elif op.kind == "Softmax":
            x = v[0] - np.max(v[0], axis=-1, keepdims=True)
            p = np.exp(x)
            result = p / np.sum(p, axis=-1, keepdims=True)
        elif op.kind == "Conv":
            x, w = v[:2]
            strides = op.attr("strides", (1, 1))
            pads = op.attr("pads", (0, 0, 0, 0))
            out_shape = model.tensor(op.outputs[0]).shape
            result = np.zeros(out_shape, dtype=np.float32)
            for batch in range(out_shape[0]):
                for oc in range(out_shape[1]):
                    for oy in range(out_shape[2]):
                        for ox in range(out_shape[3]):
                            acc = float(v[2][oc]) if len(v) == 3 else 0.0
                            for ic in range(x.shape[1]):
                                for ky in range(w.shape[2]):
                                    for kx in range(w.shape[3]):
                                        iy = oy * strides[0] + ky - pads[0]
                                        ix = ox * strides[1] + kx - pads[1]
                                        if 0 <= iy < x.shape[2] and 0 <= ix < x.shape[3]:
                                            acc += float(x[batch, ic, iy, ix]) * float(w[oc, ic, ky, kx])
                            result[batch, oc, oy, ox] = acc
        elif op.kind == "MaxPool":
            x = v[0]
            n, c, h, w = model.tensor(op.outputs[0]).shape
            result = np.empty((n, c, h, w), dtype=np.float32)
            for iy in range(h):
                for ix in range(w):
                    result[:, :, iy, ix] = x[:, :, iy*2:iy*2+2, ix*2:ix*2+2].max(axis=(2, 3))
        else:
            raise ValueError(f"No numerical reference implementation: {op.kind}")
        expected = model.tensor(op.outputs[0]).shape
        if tuple(result.shape) != expected:
            raise ValueError(f"Reference shape mismatch for {op.name}: {result.shape} != {expected}")
        tensors[op.outputs[0]] = np.asarray(result, dtype=np.float32)
    return tensors if return_all else {name: tensors[name] for name in model.outputs}
