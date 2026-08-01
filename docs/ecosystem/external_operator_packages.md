# External operator packages

External operator packages are discovered as metadata first and execute only after explicit `approved_for_reference` approval. A package returns an `ExternalOperatorDefinition`; it does not mutate compiler globals. ONNX bindings, shape/type inference, and NumPy reference behavior use versioned callback contexts. FPGAI use remains research, validation, and benchmarking; production productization follows Morfics.
