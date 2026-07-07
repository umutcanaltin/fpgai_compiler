# P1C training DDR duplicate mode fix

Problem: medium DDR CIFAR training reached HLS CSim compilation but failed because generated `deeplearn.cpp` declared `static const int FPGAI_MODE_RUN_TRAINING = 2;` twice in the same top function.

Cause: `top_train_cpp.py` has layered wrappers. The general training storage wrapper emits `FPGAI_MODE_RUN_TRAINING`. The DDR-tiled mutable wrapper also emitted the same constant.

Fix: keep `FPGAI_MODE_RUN_TRAINING` in the general training storage block and remove the duplicate declaration from the DDR-tiled mutable block. The DDR wrapper still emits `FPGAI_MODE_DDR_TILED_TRAINING = 7` and uses the existing run-training constant.

Validation target:

```bash
python -m py_compile fpgai/backends/hls/emit/top_train_cpp.py
python -m fpgai.cli compile --config examples/paper/models/medium_ddr_cifar_training.yml
```

Expected next behavior: CSim should no longer fail with redeclaration of `FPGAI_MODE_RUN_TRAINING`. If it fails later, inspect the new HLS error because that will be the next real training-backend issue.
