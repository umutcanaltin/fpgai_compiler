from fpgai.frontend.onnx import import_onnx
from fpgai.ir.passes import assign_stable_names

g = import_onnx("models/mlp.onnx", canonicalize=True, infer_shapes=True)
g = assign_stable_names(g)
print(g.summary())
