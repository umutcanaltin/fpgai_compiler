# P1A paper model suite patch

Adds paper workload coverage through the existing `fpgai/experiments/model_suite.py` path.

New generated model kinds:

- `cifar_small_cnn`: CIFAR-like 3x32x32 medium CNN.
- `large_ddr_stress_cnn`: larger 3x64x64 stress CNN for DDR/tiled validation.
- `tiny_yolo_like`: tiny detection-shaped inference workload with 4x4x7 output.

New static paper configs:

- `examples/paper/models/compact_onchip_mnist_mlp.yml`
- `examples/paper/models/compact_onchip_mnist_training.yml`
- `examples/paper/models/medium_ddr_cifar_cnn.yml`
- `examples/paper/models/medium_ddr_cifar_training.yml`
- `examples/paper/models/large_ddr_stress_cnn.yml`
- `examples/paper/models/large_ddr_yolo_like.yml`

This sprint intentionally does not run HLS/Vivado/runtime. It only creates the paper model set for static compilation and later master-result aggregation.
