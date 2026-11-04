"""Check exported trained ONNX against PyTorch outputs and Model IR (E0 only).

This is a prerequisite for experimental model provenance, NOT native ap_fixed
equivalence, Vitis HLS simulation or routed FPGA correctness.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np


def validate_exported_model(onnx_path, inputs, expected, *,
                            check_model_ir=True, atol=1e-5, rtol=1e-4):
    if (type(check_model_ir) is not bool or
        type(atol) not in (float,int) or type(rtol) not in (float,int) or
        not np.isfinite(atol) or not np.isfinite(rtol) or
        atol<0 or rtol<0):
        raise ValueError("Invalid ONNX conformance policy")
    source=Path(onnx_path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("ONNX conformance requires a regular non-linked model file")
    data=np.asarray(inputs)
    reference=np.asarray(expected)
    if data.dtype!=np.float32 or reference.dtype!=np.float32:
        raise ValueError("ONNX conformance requires exact float32 input and reference output")
    if (data.ndim<2 or not 1<=len(data)<=256 or
        reference.ndim!=2 or reference.shape[0]!=len(data) or
        not np.isfinite(data).all() or not np.isfinite(reference).all()):
        raise ValueError("Invalid or nonfinite ONNX evaluation tensors")
    original_hash=sha256(source.read_bytes()).hexdigest()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to qualify trained paper ONNX exports") from exc
    session=ort.InferenceSession(str(source),providers=["CPUExecutionProvider"])
    inputs_info=session.get_inputs()
    outputs_info=session.get_outputs()
    if len(inputs_info)!=1 or len(outputs_info)!=1:
        raise ValueError("Trained ONNX conformance requires exactly one input and output")
    name=inputs_info[0].name
    expected_shape=(1,*data.shape[1:])
    output_shape=(1,reference.shape[1])
    if (tuple(inputs_info[0].shape)!=expected_shape or
        tuple(outputs_info[0].shape)!=output_shape):
        raise ValueError("Trained ONNX input/output static geometry mismatch")
    observed=[]
    for i in range(len(data)):
        values=session.run(None,{name:data[i:i+1]})
        if (len(values)!=1 or not isinstance(values[0],np.ndarray) or
            values[0].dtype!=np.float32 or values[0].shape!=output_shape or
            not np.isfinite(values[0]).all()):
            raise ValueError("ONNX Runtime returned invalid, nonfinite or mistyped outputs")
        observed.append(values[0][0])
    onnx_values=np.stack(observed,axis=0)
    error=np.abs(onnx_values.astype(np.float64)-reference.astype(np.float64))
    bound=atol+rtol*np.abs(reference.astype(np.float64))
    if not np.all(error<=bound):
        raise ValueError("ONNX Runtime output differs numerically from trained PyTorch model")
    if (reference.shape[1]<2 or
        not np.array_equal(onnx_values.argmax(axis=1), reference.argmax(axis=1))):
        raise ValueError("ONNX Runtime class predictions disagree with trained PyTorch model")
    ir_max=None
    if check_model_ir:
        from fpgai_research.frontend.onnx import load_onnx
        from fpgai_research.validation.reference import evaluate
        model=load_onnx(str(source))
        if (len(model.inputs)!=1 or len(model.outputs)!=1 or
            model.tensor(model.inputs[0]).shape!=expected_shape or
            model.tensor(model.outputs[0]).shape!=output_shape):
            raise ValueError("Compiler Functional Model IR boundaries disagree with trained ONNX")
        ir_rows=[]
        for i in range(len(data)):
            row=evaluate(model,{model.inputs[0]:data[i:i+1]})[model.outputs[0]]
            if row.shape!=output_shape or not np.isfinite(row).all():
                raise ValueError("Model IR returned incompatible or nonfinite output")
            ir_rows.append(row[0])
        ir=np.stack(ir_rows,axis=0)
        ir_delta=np.abs(ir.astype(np.float64)-onnx_values.astype(np.float64))
        if not np.all(ir_delta<=atol+rtol*np.abs(onnx_values.astype(np.float64))):
            raise ValueError("Compiler Model IR output differs from ONNX Runtime")
        if not np.array_equal(ir.argmax(axis=1),onnx_values.argmax(axis=1)):
            raise ValueError("Compiler Model IR class predictions disagree with ONNX Runtime")
        ir_max=float(np.max(ir_delta))
    if sha256(source.read_bytes()).hexdigest()!=original_hash:
        raise ValueError("ONNX model changed during export numerical qualification")
    return {"schema":"fpgai-research/trained-onnx-conformance/v1",
            "evidence":("E0-PyTorch-ONNXRuntime-ModelIR" if check_model_ir
                        else "E0-PyTorch-ONNXRuntime-only"),
            "model_sha256":original_hash,
            "evaluated_samples":len(data),
            "onnxruntime_version":str(ort.__version__),
            "model_ir_compared":check_model_ir,
            "class_predictions_matched":True,
            "absolute_tolerance":atol,"relative_tolerance":rtol,
            "onnxruntime_max_abs_error":float(np.max(error)),
            "model_ir_max_abs_error":ir_max,
            "exact_float32_inputs":True,
            "native_vitis_executed":False,
            "vivado_executed":False}
