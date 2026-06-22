# FPGAI Example Configurations

These examples are intended to match the current public `fpgai inspect --config ...` schema.
They are examples for users and contributors, not development-only artifacts.

## Validate an example

```bash
fpgai inspect --config configs/examples/inference_compile.yml \
  --json-output build/examples/inference_inspection.json
```

If the installed console command is not available:

```bash
PYTHONPATH="$PWD" python -m fpgai.cli inspect \
  --config configs/examples/inference_compile.yml \
  --json-output build/examples/inference_inspection.json
```

## Policy

- Keep example configs small and stable.
- Prefer YAML configs for normal compilation workflows.
- Use scripts only for paper experiments collection, development utilities, or compatibility wrappers.
