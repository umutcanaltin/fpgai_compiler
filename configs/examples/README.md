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

## Dataset-backed multi-epoch training

The canonical schedule keys are:

```yaml
training:
  batch:
    size: 2
    epochs: 3
    mode: accumulated
    shuffle: true
    seed: 42
    drop_last: false
  validation:
    convergence_smoke: true
    loss_eval_records: 10
```

`training.batch.epochs` counts complete passes over the normalized dataset. Do not use the deprecated `training.execution.train_steps` as an epoch substitute; its canonical replacement, `training.batch.max_updates`, is only an explicit optimizer-update cap.

The maintained experiment template is:

```bash
PYTHONPATH=. python -m fpgai.cli sweep run \
  --config configs/sweeps/training_multi_epoch_convergence.yml \
  --out paper_experiments/training_multi_epoch_convergence
```

It inherits `examples/training/mnist_balanced10_dataset_training.yml`, which pairs `models/suite/mlp_mnist.onnx` with a balanced torchvision MNIST subset. The compiler now emits `reports/training_dataset_model_contract.json` before numerical execution and rejects mismatched model/dataset word counts or incompatible supervision. The sweep demonstrates deterministic multi-epoch learning behavior and CSim/reference artifacts; it is not by itself a convergence, generalization, or real-board result.

## Fair batch-size training ablations

FPGAI keeps two experiment contracts separate:

```bash
# Equal sample exposure: epochs/record visits are held constant; update counts differ.
PYTHONPATH=. python -m fpgai.cli sweep run \
  --config configs/sweeps/training_multi_epoch_convergence.yml \
  --out paper_experiments/training_batch_equal_exposure

# Equal optimizer budget: six updates are held constant; record exposure differs.
PYTHONPATH=. python -m fpgai.cli sweep run \
  --config configs/sweeps/training_batch_equal_update_budget.yml \
  --out paper_experiments/training_batch_equal_updates6
```

Both sweeps emit `training_learning_ablation_summary.csv`,
`training_learning_ablation_summary.md`, and `training_claim_eligibility.json`.
The summary records sample visits and optimizer updates side by side so a batch-size
claim cannot silently mix those two experimental budgets.

### Strict training batch ablations

- `configs/sweeps/training_batch_equal_exposure_strict3.yml` compares batch sizes 2, 5, and 10 at the same three-epoch / 30-record-visit exposure.
- `configs/sweeps/training_batch_equal_update_budget.yml` compares the same batch sizes at six optimizer updates with intentionally different record exposure.
- Generated reports expose control roles, normalized loss reduction, seed/replicate counts, and statistical claim eligibility.
