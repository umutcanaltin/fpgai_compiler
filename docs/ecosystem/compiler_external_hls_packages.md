# Compiler wiring for external HLS packages

FPGAI can compile a single external ONNX operator into a generated Vitis HLS project when `ecosystem.enabled` is true. Package discovery remains metadata-only; Python operator code is loaded only for package IDs explicitly enabled and granted `approved_for_reference` trust.

The E4B path performs discovery, external ONNX import, implementation compatibility filtering, explainable selection, package locking, and generic HLS project generation. It does not yet splice external implementations into mixed built-in/external graphs.

```yaml
ecosystem:
  enabled: true
  project_root: .
  package_directories: [examples/packages]
  operator_packages:
    enable: [community.scale_bias_operator]
  trust:
    community.scale_bias_operator: approved_for_reference

implementations:
  enable: [community.scale_bias_hls]
  selection_policy: balanced
  operators:
    community.operator.scale_bias:
      preferred: [community.scale_bias_hls]
      allow_fallback: false
```

Generated traceability artifacts include `package-lock.yml`, package discovery reports, external operator loading status, implementation selection reports, the external HLS integration report, and `manifest.json`.
