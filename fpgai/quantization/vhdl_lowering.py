from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fpgai.quantization.contracts import QuantizationParameters, quantization_spec_from_mapping
from fpgai.quantization.hardware import derive_requantization_contract


def emit_quantized_add_int8x4_vhdl_package(
    package_root: str | Path,
    partition: Mapping[str, Any],
) -> Path:
    """Emit a PTQ-specialized packed-int8x4 VHDL Add package."""
    root = Path(package_root)
    rtl = root / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)

    add = partition.get("add") if isinstance(partition.get("add"), Mapping) else None
    if not isinstance(add, Mapping):
        raise ValueError("QVHDL001: residual partition is missing Add metadata")
    lowering = add.get("lowering") if isinstance(add.get("lowering"), Mapping) else None
    if not isinstance(lowering, Mapping):
        raise ValueError("QVHDL002: residual Add partition is missing quantized lowering metadata")

    for key in ("left_quantization", "right_quantization", "output_quantization"):
        q = add.get(key)
        if not isinstance(q, Mapping):
            raise ValueError(f"QVHDL003: residual Add partition is missing {key}")
        spec = q.get("spec") if isinstance(q.get("spec"), Mapping) else {}
        if int(spec.get("bits", 0)) != 8 or str(spec.get("granularity", "")) != "per_tensor":
            raise ValueError("QVHDL004: packed VHDL Add currently requires per-tensor int8 contracts")

    out_spec = add["output_quantization"]["spec"]
    if str(out_spec.get("saturation", "saturate")) != "saturate":
        raise ValueError("QVHDL005: packed VHDL Add currently requires saturation=saturate")
    if str(out_spec.get("rounding", "nearest")) not in {"nearest", "floor", "ceil"}:
        raise ValueError("QVHDL006: packed VHDL Add received an unsupported rounding policy")

    values = {
        "left_zero": int(lowering.get("left_zero", 0)),
        "left_multiplier": int(lowering["left_multiplier"]),
        "left_shift": int(lowering["left_shift"]),
        "right_zero": int(lowering.get("right_zero", 0)),
        "right_multiplier": int(lowering["right_multiplier"]),
        "right_shift": int(lowering["right_shift"]),
        "output_zero": int(lowering.get("output_zero", 0)),
        "qmin": int(lowering.get("qmin", -128)),
        "qmax": int(lowering.get("qmax", 127)),
        "rounding_mode": int(lowering.get("rounding_mode", 0)),
    }
    if values["qmin"] != -128 or values["qmax"] != 127:
        raise ValueError("QVHDL007: packed VHDL Add currently requires signed int8 output range")

    source = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity quantized_add_int8x4_vhdl is
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;
    input_valid  : in  std_logic;
    input_ready  : out std_logic;
    left_data    : in  signed(31 downto 0);
    right_data   : in  signed(31 downto 0);
    output_valid : out std_logic;
    output_ready : in  std_logic;
    output_data  : out signed(31 downto 0)
  );
end entity;

architecture rtl of quantized_add_int8x4_vhdl is
  constant LEFT_ZERO        : integer := {values['left_zero']};
  constant LEFT_MULTIPLIER  : integer := {values['left_multiplier']};
  constant LEFT_SHIFT       : natural := {values['left_shift']};
  constant RIGHT_ZERO       : integer := {values['right_zero']};
  constant RIGHT_MULTIPLIER : integer := {values['right_multiplier']};
  constant RIGHT_SHIFT      : natural := {values['right_shift']};
  constant OUTPUT_ZERO      : integer := {values['output_zero']};
  constant QMIN             : integer := {values['qmin']};
  constant QMAX             : integer := {values['qmax']};
  constant ROUNDING_MODE    : integer := {values['rounding_mode']};

  signal valid_q : std_logic := '0';
  signal data_q  : signed(31 downto 0) := (others => '0');

  function requant_lane(
    raw        : signed(7 downto 0);
    zero_point : integer;
    multiplier : integer;
    shift      : natural
  ) return signed is
    variable centered  : signed(31 downto 0);
    variable product   : signed(63 downto 0);
    variable rounded   : signed(63 downto 0);
    variable magnitude : signed(63 downto 0);
    variable half      : signed(63 downto 0);
    variable shifted   : signed(63 downto 0);
    variable with_zero : signed(63 downto 0);
  begin
    centered := resize(raw, 32) - to_signed(zero_point, 32);
    product := centered * to_signed(multiplier, 32);
    rounded := product;
    shifted := product;
    if shift > 0 then
      half := shift_left(to_signed(1, 64), shift - 1);
      if ROUNDING_MODE = 0 then
        if product >= 0 then
          rounded := product + half;
          shifted := shift_right(rounded, shift);
        else
          magnitude := -product;
          rounded := magnitude + half;
          shifted := -shift_right(rounded, shift);
        end if;
      elsif ROUNDING_MODE = 1 then
        shifted := shift_right(product, shift);
      else
        shifted := -shift_right(-product, shift);
      end if;
    end if;
    with_zero := shifted + to_signed(OUTPUT_ZERO, 64);
    if with_zero < to_signed(QMIN, 64) then
      return to_signed(QMIN, 8);
    elsif with_zero > to_signed(QMAX, 64) then
      return to_signed(QMAX, 8);
    end if;
    return resize(with_zero, 8);
  end function;

  function add_lane(left_raw, right_raw : signed(7 downto 0)) return signed is
    variable left_q  : signed(7 downto 0);
    variable right_q : signed(7 downto 0);
    variable sum_v   : integer;
  begin
    left_q := requant_lane(left_raw, LEFT_ZERO, LEFT_MULTIPLIER, LEFT_SHIFT);
    right_q := requant_lane(right_raw, RIGHT_ZERO, RIGHT_MULTIPLIER, RIGHT_SHIFT);
    sum_v := to_integer(left_q) + to_integer(right_q) - OUTPUT_ZERO;
    if sum_v < QMIN then
      return to_signed(QMIN, 8);
    elsif sum_v > QMAX then
      return to_signed(QMAX, 8);
    end if;
    return to_signed(sum_v, 8);
  end function;
begin
  input_ready  <= (not valid_q) or output_ready;
  output_valid <= valid_q;
  output_data  <= data_q;

  process(clk)
    variable next_data : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        valid_q <= '0';
        data_q  <= (others => '0');
      elsif ((not valid_q) or output_ready) = '1' then
        valid_q <= input_valid;
        if input_valid = '1' then
          next_data := (others => '0');
          for lane in 0 to 3 loop
            next_data((lane + 1) * 8 - 1 downto lane * 8) := add_lane(
              left_data((lane + 1) * 8 - 1 downto lane * 8),
              right_data((lane + 1) * 8 - 1 downto lane * 8)
            );
          end loop;
          data_q <= next_data;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""
    (rtl / "quantized_add_int8x4_vhdl.vhd").write_text(source, encoding="utf-8")

    manifest = """schema: fpgai.package/v1
package:
  id: generated.quantized_add_int8x4_vhdl
  name: Generated Quantized Int8x4 Add VHDL
  version: 1.0.0
  asset_type: implementation
  provider: project_local
  description: PTQ-specialized two-input packed int8 VHDL residual Add implementation.
usage:
  platform_scope: research
  permitted_uses: [research, experimentation, validation, benchmarking, education]
  production_path: morfics
license:
  category: open_source
  identifier: Apache-2.0
compatibility:
  fpgai_contract: ">=1.0,<2.0"
  boards: [kv260, kr260]
  precisions: [int8]
  toolchains:
    - name: vivado
      versions: ["2023.2"]
capabilities:
  inference: true
  training:
    forward: true
    backward_input: false
    parameter_gradients: false
    bias_gradients: false
    optimizer_update: false
entrypoints:
  implementation:
    language: vhdl
    backend: vhdl
    top: quantized_add_int8x4_vhdl
    sources: [rtl/quantized_add_int8x4_vhdl.vhd]
    source_order: [rtl/quantized_add_int8x4_vhdl.vhd]
implementation:
  operator_id: onnx.add
  backend: vhdl
integration:
  vhdl:
    abi: tensor_ports_ready_valid_v1
    data_width: 32
    signed: false
    handshake_policy: grouped_transaction
    inputs:
      - name: left
        data: left_data
      - name: right
        data: right_data
    outputs:
      - name: output
        data: output_data
interfaces:
  left:
    direction: input
    protocol: axi_stream
    data_type: packed_int8x4
  right:
    direction: input
    protocol: axi_stream
    data_type: packed_int8x4
  output:
    direction: output
    protocol: axi_stream
    data_type: packed_int8x4
validation:
  declared_level: unvalidated
metrics:
  latency_cycles: 1
  initiation_interval: 1
"""
    (root / "fpgai.yaml").write_text(manifest, encoding="utf-8")
    return root


def emit_quantized_relu_int8x4_vhdl_package(
    package_root: str | Path,
    partition: Mapping[str, Any],
) -> Path:
    """Emit a packed-int8x4 VHDL ReLU specialized to compiler requantization semantics."""
    root = Path(package_root)
    rtl = root / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)

    relu = partition.get("relu") if isinstance(partition.get("relu"), Mapping) else None
    if not isinstance(relu, Mapping):
        raise ValueError("QVHDL101: residual partition is missing ReLU metadata")
    lowering = relu.get("lowering") if isinstance(relu.get("lowering"), Mapping) else None

    for key in ("input_quantization", "output_quantization"):
        q = relu.get(key)
        if not isinstance(q, Mapping):
            raise ValueError(f"QVHDL103: residual ReLU partition is missing {key}")
        spec = q.get("spec") if isinstance(q.get("spec"), Mapping) else {}
        if int(spec.get("bits", 0)) != 8 or str(spec.get("granularity", "")) != "per_tensor":
            raise ValueError("QVHDL104: packed VHDL ReLU currently requires per-tensor int8 contracts")


    if not isinstance(lowering, Mapping):
        # Backward-compatible artifact handling: older partition reports contain
        # complete quantization contracts but predate explicit ReLU lowering.
        def _params(raw: Mapping[str, Any], *, path: str) -> QuantizationParameters:
            spec_raw = raw.get("spec") if isinstance(raw.get("spec"), Mapping) else None
            if not isinstance(spec_raw, Mapping):
                raise ValueError(f"QVHDL102: {path} is missing quantization spec")
            spec = quantization_spec_from_mapping(spec_raw, path=f"{path}.spec")

            def _value(name: str, default: Any) -> Any:
                value = raw.get(name, default)
                return tuple(value) if isinstance(value, list) else value

            return QuantizationParameters(
                spec=spec,
                scale=_value("scale", 1.0),
                zero_point=_value("zero_point", 0),
                observed_min=_value("observed_min", 0.0),
                observed_max=_value("observed_max", 0.0),
            )

        input_q = _params(relu["input_quantization"], path="relu.input_quantization")
        output_q = _params(relu["output_quantization"], path="relu.output_quantization")
        contract = derive_requantization_contract(input_q, output_q)
        rounding_codes = {"nearest": 0, "floor": 1, "ceil": 2}
        saturation_codes = {"saturate": 0, "wrap": 1}
        lowering = {
            "input_zero": int(input_q.zero_point),
            "multiplier": int(contract.multiplier),
            "shift": int(contract.shift),
            "output_zero": int(output_q.zero_point),
            "qmin": int(output_q.spec.qmin),
            "qmax": int(output_q.spec.qmax),
            "rounding_mode": int(rounding_codes[output_q.spec.rounding]),
            "saturation_mode": int(saturation_codes[output_q.spec.saturation]),
        }

    out_spec = relu["output_quantization"]["spec"]
    if str(out_spec.get("saturation", "saturate")) != "saturate":
        raise ValueError("QVHDL105: packed VHDL ReLU currently requires saturation=saturate")

    values = {
        "input_zero": int(lowering.get("input_zero", 0)),
        "multiplier": int(lowering["multiplier"]),
        "shift": int(lowering["shift"]),
        "output_zero": int(lowering.get("output_zero", 0)),
        "qmin": int(lowering.get("qmin", -128)),
        "qmax": int(lowering.get("qmax", 127)),
        "rounding_mode": int(lowering.get("rounding_mode", 0)),
    }
    if values["qmin"] != -128 or values["qmax"] != 127:
        raise ValueError("QVHDL106: packed VHDL ReLU currently requires signed int8 output range")

    source = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity quantized_relu_int8x4_vhdl is
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;
    input_valid  : in  std_logic;
    input_ready  : out std_logic;
    input_data   : in  signed(31 downto 0);
    output_valid : out std_logic;
    output_ready : in  std_logic;
    output_data  : out signed(31 downto 0)
  );
end entity;

architecture rtl of quantized_relu_int8x4_vhdl is
  constant INPUT_ZERO    : integer := {values['input_zero']};
  constant MULTIPLIER    : integer := {values['multiplier']};
  constant SHIFT         : natural := {values['shift']};
  constant OUTPUT_ZERO   : integer := {values['output_zero']};
  constant QMIN          : integer := {values['qmin']};
  constant QMAX          : integer := {values['qmax']};
  constant ROUNDING_MODE : integer := {values['rounding_mode']};

  signal valid_q : std_logic := '0';
  signal data_q  : signed(31 downto 0) := (others => '0');

  function relu_lane(raw : signed(7 downto 0)) return signed is
    variable centered  : signed(31 downto 0);
    variable product   : signed(63 downto 0);
    variable rounded   : signed(63 downto 0);
    variable magnitude : signed(63 downto 0);
    variable half      : signed(63 downto 0);
    variable shifted   : signed(63 downto 0);
    variable with_zero : signed(63 downto 0);
  begin
    if to_integer(raw) < INPUT_ZERO then
      centered := (others => '0');
    else
      centered := resize(raw, 32) - to_signed(INPUT_ZERO, 32);
    end if;
    product := centered * to_signed(MULTIPLIER, 32);
    rounded := product;
    shifted := product;
    if SHIFT > 0 then
      half := shift_left(to_signed(1, 64), SHIFT - 1);
      if ROUNDING_MODE = 0 then
        if product >= 0 then
          rounded := product + half;
          shifted := shift_right(rounded, SHIFT);
        else
          magnitude := -product;
          rounded := magnitude + half;
          shifted := -shift_right(rounded, SHIFT);
        end if;
      elsif ROUNDING_MODE = 1 then
        shifted := shift_right(product, SHIFT);
      else
        shifted := -shift_right(-product, SHIFT);
      end if;
    end if;
    with_zero := shifted + to_signed(OUTPUT_ZERO, 64);
    if with_zero < to_signed(QMIN, 64) then
      return to_signed(QMIN, 8);
    elsif with_zero > to_signed(QMAX, 64) then
      return to_signed(QMAX, 8);
    end if;
    return resize(with_zero, 8);
  end function;
begin
  input_ready  <= (not valid_q) or output_ready;
  output_valid <= valid_q;
  output_data  <= data_q;

  process(clk)
    variable next_data : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        valid_q <= '0';
        data_q  <= (others => '0');
      elsif ((not valid_q) or output_ready) = '1' then
        valid_q <= input_valid;
        if input_valid = '1' then
          next_data := (others => '0');
          for lane in 0 to 3 loop
            next_data((lane + 1) * 8 - 1 downto lane * 8) :=
              relu_lane(input_data((lane + 1) * 8 - 1 downto lane * 8));
          end loop;
          data_q <= next_data;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""
    (rtl / "quantized_relu_int8x4_vhdl.vhd").write_text(source, encoding="utf-8")

    manifest = """schema: fpgai.package/v1
package:
  id: generated.quantized_relu_int8x4_vhdl
  name: Generated Quantized Int8x4 ReLU VHDL
  version: 1.0.0
  asset_type: implementation
  provider: project_local
  description: Compiler-specialized packed int8 VHDL ReLU with requantization.
usage:
  platform_scope: research
  permitted_uses: [research, experimentation, validation, benchmarking, education]
  production_path: morfics
license:
  category: open_source
  identifier: Apache-2.0
compatibility:
  fpgai_contract: ">=1.0,<2.0"
  boards: [kv260, kr260]
  precisions: [int8]
  toolchains:
    - name: vivado
      versions: ["2023.2"]
capabilities:
  inference: true
  training:
    forward: true
    backward_input: false
    parameter_gradients: false
    bias_gradients: false
    optimizer_update: false
entrypoints:
  implementation:
    language: vhdl
    backend: vhdl
    top: quantized_relu_int8x4_vhdl
    sources: [rtl/quantized_relu_int8x4_vhdl.vhd]
    source_order: [rtl/quantized_relu_int8x4_vhdl.vhd]
implementation:
  operator_id: onnx.relu
  backend: vhdl
integration:
  vhdl:
    abi: tensor_ports_ready_valid_v1
    data_width: 32
    signed: false
    handshake_policy: grouped_transaction
    inputs:
      - name: input
        data: input_data
    outputs:
      - name: output
        data: output_data
interfaces:
  input:
    direction: input
    protocol: axi_stream
    data_type: packed_int8x4
  output:
    direction: output
    protocol: axi_stream
    data_type: packed_int8x4
validation:
  declared_level: unvalidated
metrics:
  latency_cycles: 1
  initiation_interval: 1
"""
    (root / "fpgai.yaml").write_text(manifest, encoding="utf-8")
    return root
