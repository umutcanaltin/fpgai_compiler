# `tensor_ports_v1` HLS ABI

`tensor_ports_v1` is the first FPGAI external-HLS ABI that supports more than one runtime tensor port. Package manifests declare ordered `inputs` and `outputs` under `integration.hls`. The initial v1 profile uses one scalar C++ type and one shared element count, so all runtime tensor ports must currently have equal flattened sizes. This restriction is explicit and will be relaxed by later shape-aware ABI versions.

The ABI is intended for merge and multi-input operators such as Add and custom residual operators. It is separate from the older unary `flat_array_v1` ABI, which remains supported.
