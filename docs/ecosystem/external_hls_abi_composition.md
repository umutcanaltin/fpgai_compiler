# External HLS ABI composition

`flat_array_v1` calls a package function with input pointer, output pointer, element count, and ordered scalar attributes. E4C emits explicit conversion buffers when package scalar types differ from FPGAI activation aliases; pointer reinterpretation is not used. Package sources and headers are copied into namespaced directories under the normal generated HLS project and are never edited in place.

Later ABI versions will cover multiple tensors, streaming protocols, parameters, state, gradients, and training callbacks.
