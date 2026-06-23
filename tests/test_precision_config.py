from __future__ import annotations

from pathlib import Path

import pytest

from fpgai.config.loader import (
    ConfigError,
    load_config,
)


def _base_config(
    model_path: Path,
    extra: str = "",
) -> str:
    return f"""
version: 1

model:
  path: {model_path}

pipeline:
  mode: inference

operators:
  supported:
    - Dense
    - Relu

numerics:
  defaults:
    activation:
      type: ap_fixed
      total_bits: 16
      int_bits: 6

    weight:
      type: ap_fixed
      total_bits: 16
      int_bits: 6

    bias:
      type: ap_fixed
      total_bits: 24
      int_bits: 10

    accum:
      type: ap_fixed
      total_bits: 24
      int_bits: 10

{extra}
"""


def _write_config(
    tmp_path: Path,
    content: str,
) -> Path:
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    config_path = tmp_path / "configs/examples/default_compile.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        content.replace(
            "__MODEL__",
            str(model_path),
        ),
        encoding="utf-8",
    )

    return config_path


def test_valid_precision_config_loads(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    config_path = tmp_path / "configs/examples/default_compile.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        _base_config(
            model_path,
            """
analysis:
  quantization_report:
    enabled: true
    seed: 0

  precision_sweep:
    enabled: true
    layer_overrides: clear
    require_prediction_match: true
    minimum_cosine: 0.99

    candidates:
      - name: fx8_3
        defaults:
          activation:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          weight:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          bias:
            type: ap_fixed
            total_bits: 16
            int_bits: 6

          accum:
            type: ap_fixed
            total_bits: 16
            int_bits: 6
""",
        ),
        encoding="utf-8",
    )

    config = load_config(
        str(config_path)
    )

    assert config.pipeline.mode == "inference"

    sweep = config.raw[
        "analysis"
    ]["precision_sweep"]

    assert sweep["layer_overrides"] == "clear"
    assert sweep["minimum_cosine"] == 0.99


def test_config_rejects_boolean_bit_width(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  defaults:
    activation:
      type: ap_fixed
      total_bits: true
      int_bits: 6
""",
    )

    with pytest.raises(
        ConfigError,
        match="numerics.defaults.activation",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_fractional_bit_width(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  defaults:
    activation:
      type: ap_fixed
      total_bits: 16.0
      int_bits: 6
""",
    )

    with pytest.raises(
        ConfigError,
        match="numerics.defaults.activation",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_int_bits_larger_than_total_bits(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  defaults:
    activation:
      type: ap_fixed
      total_bits: 8
      int_bits: 9
""",
    )

    with pytest.raises(
        ConfigError,
        match="numerics.defaults.activation",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_unknown_default_numeric_role(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  defaults:
    activation:
      type: ap_fixed
      total_bits: 16
      int_bits: 6

    weights:
      type: ap_fixed
      total_bits: 16
      int_bits: 6
""",
    )

    with pytest.raises(
        ConfigError,
        match="Unknown numeric role",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_empty_layer_match(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  layers:
    - match: {}
      weight:
        type: ap_fixed
        total_bits: 8
        int_bits: 3
""",
    )

    with pytest.raises(
        ConfigError,
        match="At least one of name, op_type, or index",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_unknown_layer_match_field(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  layers:
    - match:
        layer: dense0

      weight:
        type: ap_fixed
        total_bits: 8
        int_bits: 3
""",
    )

    with pytest.raises(
        ConfigError,
        match="Unknown match field",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_negative_layer_index(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

numerics:
  layers:
    - match:
        index: -1

      weight:
        type: ap_fixed
        total_bits: 8
        int_bits: 3
""",
    )

    with pytest.raises(
        ConfigError,
        match="non-negative integer",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_duplicate_operator_names(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense
    - Dense
""",
    )

    with pytest.raises(
        ConfigError,
        match="Duplicate operator names",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_duplicate_sweep_names(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

analysis:
  precision_sweep:
    enabled: true

    candidates:
      - name: same
        defaults:
          activation:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          weight:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          bias:
            type: ap_fixed
            total_bits: 16
            int_bits: 6

          accum:
            type: ap_fixed
            total_bits: 16
            int_bits: 6

      - name: same
        defaults:
          activation:
            type: ap_fixed
            total_bits: 10
            int_bits: 4

          weight:
            type: ap_fixed
            total_bits: 10
            int_bits: 4

          bias:
            type: ap_fixed
            total_bits: 18
            int_bits: 7

          accum:
            type: ap_fixed
            total_bits: 20
            int_bits: 8
""",
    )

    with pytest.raises(
        ConfigError,
        match="Duplicate precision sweep candidate name",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_invalid_sweep_override_mode(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

analysis:
  precision_sweep:
    enabled: false
    layer_overrides: sometimes
""",
    )

    with pytest.raises(
        ConfigError,
        match="layer_overrides",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_invalid_candidate_override_mode(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

analysis:
  precision_sweep:
    enabled: true
    layer_overrides: clear

    candidates:
      - name: fx8
        layer_overrides: invalid

        defaults:
          activation:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          weight:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          bias:
            type: ap_fixed
            total_bits: 16
            int_bits: 6

          accum:
            type: ap_fixed
            total_bits: 16
            int_bits: 6
""",
    )

    with pytest.raises(
        ConfigError,
        match="layer_overrides",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_missing_candidate_role(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

analysis:
  precision_sweep:
    enabled: true

    candidates:
      - name: incomplete
        defaults:
          activation:
            type: ap_fixed
            total_bits: 8
            int_bits: 3

          weight:
            type: ap_fixed
            total_bits: 8
            int_bits: 3
""",
    )

    with pytest.raises(
        ConfigError,
        match="Missing precision specification",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_invalid_minimum_cosine(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

analysis:
  precision_sweep:
    enabled: false
    minimum_cosine: 1.5
""",
    )

    with pytest.raises(
        ConfigError,
        match="minimum_cosine",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_duplicate_training_aliases(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: training_on_device

operators:
  supported:
    - Dense

numerics:
  training:
    grad:
      type: ap_fixed
      total_bits: 24
      int_bits: 10

    gradient:
      type: ap_fixed
      total_bits: 20
      int_bits: 8
""",
    )

    with pytest.raises(
        ConfigError,
        match="Duplicate alias",
    ):
        load_config(
            str(config_path)
        )


def test_config_rejects_duplicate_yaml_keys(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
version: 1

model:
  path: __MODEL__

pipeline:
  mode: inference

operators:
  supported:
    - Dense

analysis:
  precision_sweep:
    enabled: false
    enabled: true
""",
    )

    with pytest.raises(
        ConfigError,
        match="duplicate key 'enabled'",
    ):
        load_config(
            str(config_path)
        )
