## Where to add features

### New ONNX operator support
- Map to canonical ops in: `fpgai/frontend/onnx/canonicalize.py`
- Add fusions in: `fpgai/frontend/onnx/patterns.py`

### New compiler graph transformation
- Add a pass in: `fpgai/ir/passes/`
- Wire it in: `fpgai/engine/compiler.py`

### New backend
- Create a backend under: `fpgai/backends/<name>/`
- Call it from: `fpgai/engine/compiler.py`
