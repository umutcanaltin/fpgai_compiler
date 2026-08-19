from __future__ import annotations

"""Branch-aware inference top emission using FPGAI tensor liveness."""

from typing import Any, Mapping

from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.types_h import _default_precision, _op_precision_from_attrs, _spec_bits, _spec_to_ap, _tensor_cpp_type, _cpp_type_bits
from fpgai.ir.tensor_ops import axis_geometry, normalize_axis, resolve_resize_shape, resolve_slice_spec
from fpgai.engine.network_execution import requested_network_execution_mode


def _flat_size(shape: Any) -> int:
    dims = tuple(int(x) for x in (shape or (1,)))
    if len(dims) > 1 and dims[0] == 1:
        dims = dims[1:]
    total = 1
    for dim in dims:
        if dim <= 0:
            raise ValueError("HLSDAG001: dynamic/non-positive tensor dimension")
        total *= dim
    return total


def _tensor_words(graph: Any, name: str) -> int:
    spec = graph.get_tensor(name) if hasattr(graph, "get_tensor") else None
    if spec is None or not getattr(spec, "shape", None):
        raise ValueError(f"HLSDAG002: missing static shape for tensor {name!r}")
    return _flat_size(spec.shape)


def _resolved_tensor_types(graph: Any, raw_cfg: Mapping[str, Any] | None) -> tuple[dict[str, str], str]:
    raw = dict(raw_cfg or {})
    defaults = _default_precision(raw)
    activation_default = _spec_to_ap(defaults["activation"])
    types: dict[str, str] = {}
    for name, spec in (getattr(graph, "tensors", {}) or {}).items():
        types[str(name)] = _tensor_cpp_type(spec, activation_default)
    for name in getattr(graph, "inputs", []) or []:
        spec = graph.get_tensor(str(name)) if hasattr(graph, "get_tensor") else None
        types[str(name)] = _tensor_cpp_type(spec, activation_default)
    for op in getattr(graph, "ops", []) or []:
        precision = _op_precision_from_attrs(op, defaults)
        output_type = _spec_to_ap(precision["activation"])
        for name in getattr(op, "outputs", []) or []:
            spec = graph.get_tensor(str(name)) if hasattr(graph, "get_tensor") else None
            types[str(name)] = _tensor_cpp_type(spec, output_type)
    return types, _spec_to_ap(defaults["accum"])




def _control_interface_pragma(raw_cfg: Mapping[str, Any] | None) -> str:
    raw = dict(raw_cfg or {})
    targets = raw.get("targets") if isinstance(raw.get("targets"), Mapping) else {}
    hls = targets.get("hls") if isinstance(targets, Mapping) and isinstance(targets.get("hls"), Mapping) else {}
    protocol = str(hls.get("control_protocol", "s_axilite")).strip().lower()
    if protocol == "s_axilite":
        return "#pragma HLS INTERFACE s_axilite port=return bundle=control"
    if protocol == "ap_ctrl_none":
        return "#pragma HLS INTERFACE ap_ctrl_none port=return"
    raise ValueError(
        f"HLSDAG006: targets.hls.control_protocol must be s_axilite or ap_ctrl_none, got {protocol!r}"
    )

def _cfg_get(raw: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    current: Any = raw if isinstance(raw, Mapping) else {}
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _io_movement_kind(raw_cfg: Mapping[str, Any] | None, role: str) -> str:
    role = str(role).strip().lower()
    if role == "input":
        prefixes = ("data_movement.inputs.import", "data_movement.inputs", "data_movement.input.load")
    elif role == "output":
        prefixes = ("data_movement.outputs.export", "data_movement.outputs", "data_movement.output.store")
    else:
        raise ValueError(f"HLSDAG090: unknown I/O role {role!r}")

    interface = ""
    transport = ""
    policy = ""
    tiled_enabled = False
    for prefix in prefixes:
        if not interface:
            interface = str(_cfg_get(raw_cfg, f"{prefix}.interface", "") or "").strip().lower().replace("-", "_")
        if not transport:
            transport = str(_cfg_get(raw_cfg, f"{prefix}.transport", "") or "").strip().lower().replace("-", "_")
        if not policy:
            policy = str(_cfg_get(raw_cfg, f"{prefix}.policy", "") or "").strip().lower().replace("-", "_")
        tiled = _cfg_get(raw_cfg, f"{prefix}.tiled", None)
        if isinstance(tiled, Mapping):
            tiled_enabled = tiled_enabled or bool(tiled.get("enabled", False))
        elif tiled is not None:
            tiled_enabled = tiled_enabled or bool(tiled)

    if tiled_enabled and policy in {"", "full"}:
        policy = "tiled"
    if interface in {"m_axi", "maxi", "ddr"}:
        return "m_axi_tiled" if policy == "tiled" else "m_axi_full"
    if interface in {"axi_stream", "axis", "stream"} or transport in {"dma", "axi_dma"}:
        return "axi_stream_tiled" if policy == "tiled" else "axi_stream_full"
    return "axi_stream_tiled" if policy == "tiled" else "axi_stream_full"


def _io_tile_size(raw_cfg: Mapping[str, Any] | None, role: str, words: int) -> int:
    if role == "input":
        prefixes = ("data_movement.inputs.import", "data_movement.inputs", "data_movement.input.load")
    else:
        prefixes = ("data_movement.outputs.export", "data_movement.outputs", "data_movement.output.store")
    value: Any = None
    for prefix in prefixes:
        if value is None:
            value = _cfg_get(raw_cfg, f"{prefix}.tile_size", None)
        tiled = _cfg_get(raw_cfg, f"{prefix}.tiled", None)
        if value is None and isinstance(tiled, Mapping):
            value = tiled.get("tile_size", tiled.get("size", tiled.get("words")))
    try:
        parsed = int(value) if value is not None else min(64, max(1, int(words)))
    except (TypeError, ValueError):
        parsed = min(64, max(1, int(words)))
    return min(max(1, parsed), max(1, int(words)))


def _resize_mode_codes(attrs: Mapping[str, Any]) -> tuple[int, int]:
    coord = str(attrs.get("coordinate_transformation_mode", "asymmetric")).strip().lower()
    nearest = str(attrs.get("nearest_mode", "floor")).strip().lower()
    coord_codes = {"asymmetric": 0, "half_pixel": 1, "align_corners": 2}
    nearest_codes = {"floor": 0, "round_prefer_floor": 1, "ceil": 2, "round_prefer_ceil": 3}
    if coord not in coord_codes:
        raise RuntimeError(f"HLSDAG054: unsupported nearest Resize coordinate_transformation_mode {coord!r}")
    if nearest not in nearest_codes:
        raise RuntimeError(f"HLSDAG054: unsupported nearest Resize nearest_mode {nearest!r}")
    return coord_codes[coord], nearest_codes[nearest]



def _rewrite_external_state_signature(source: str, *, top_name: str, slots: list[Mapping[str, Any]]) -> str:
    """Expose persistent DDR/host state as explicit m_axi ports.

    On-chip BRAM/URAM state remains static local storage. External state keeps the
    same tensor symbol used by state operators, but the symbol becomes a top-level
    pointer so runtime allocation and generated hardware refer to one physical buffer.
    """
    if not slots:
        return source
    import re
    pattern = re.compile(r'(extern "C" void ' + re.escape(top_name) + r'\()(?P<args>.*?)(\) \{)', re.DOTALL)
    match = pattern.search(source)
    if match is None:
        raise ValueError("HLSDAG096: could not locate DAG top signature for external persistent state")
    args = match.group('args').strip()
    additions: list[str] = []
    for slot in slots:
        additions.append(f"{slot['cpp_type']}* {slot['name']}")
        additions.append(f"int* {slot['name']}_cursor")
    merged = args + (", " if args else "") + ", ".join(additions)
    source = source[:match.start('args')] + merged + source[match.end('args'):]
    marker = source.find("{", match.start())
    if marker < 0:
        raise ValueError("HLSDAG097: malformed DAG top while adding external persistent state")
    pragmas = []
    for index, slot in enumerate(slots):
        name = str(slot['name'])
        pragmas.extend([
            f"#pragma HLS INTERFACE m_axi port={name} offset=slave bundle=gmem_state_{index}",
            f"#pragma HLS INTERFACE s_axilite port={name} bundle=control",
            f"#pragma HLS INTERFACE m_axi port={name}_cursor offset=slave bundle=gmem_state_cursor_{index}",
            f"#pragma HLS INTERFACE s_axilite port={name}_cursor bundle=control",
            f"    // FPGAI_EXTERNAL_PERSISTENT_STATE tensor={','.join(str(x) for x in slot.get('tensors', []))} storage={slot.get('storage')} words={slot.get('words')} cursor={name}_cursor[0]",
        ])
    return source[:marker+1] + "\n" + "\n".join(pragmas) + source[marker+1:]


def _tied_parameter_bindings(graph: Any) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    runtime = dict(getattr(getattr(graph, 'semantics', None), 'runtime_contract', {}) or {})
    session = dict(runtime.get('autoregressive_session', {}) or {})
    groups = list(session.get('tied_parameter_groups', []) or [])
    constants = getattr(graph, 'constants', {}) or {}
    bindings: dict[str, dict[str, str]] = {}
    declarations: list[dict[str, Any]] = []
    import numpy as np
    for index, group in enumerate(groups):
        members = list((group or {}).get('members', []) or [])
        if not members:
            continue
        canonical = str(members[0].get('tensor') or '')
        if canonical not in constants:
            continue
        symbol = f"fpgai_tied_parameter_{index}"
        arr = np.asarray(constants[canonical], dtype=float).reshape(-1)
        declarations.append({'symbol': symbol, 'tensor': canonical, 'values': arr})
        canonical_shape = tuple(int(x) for x in graph.get_tensor(canonical).shape)
        for member in members:
            name = str(member.get('tensor') or '')
            view = str(member.get('view') or 'native').lower()
            if name not in constants:
                continue
            member_shape = tuple(int(x) for x in graph.get_tensor(name).shape)
            expected = canonical_shape if view == 'native' else tuple(reversed(canonical_shape))
            if member_shape != expected:
                raise ValueError(f"HLSDAG098: tied parameter {name!r} shape {member_shape} is incompatible with {view} view of canonical {canonical!r} {canonical_shape}")
            bindings[name] = {'symbol': symbol, 'view': view, 'canonical': canonical}
    return bindings, declarations


def emit_dag_top_cpp(
    graph: Any,
    *,
    top_name: str,
    weights_mode: str,
    raw_cfg: Mapping[str, Any] | None = None,
    external_composition_plan: Any = None,
    tensor_liveness: Mapping[str, Any] | None = None,
    buffer_allocation: Mapping[str, Any] | None = None,
) -> str:
    normalized_weights_mode = str(weights_mode).strip().lower()
    supported_weight_modes = {"embedded", "stream", "streamed", "ddr", "dma_ddr", "uram", "ddr_tiled"}
    if normalized_weights_mode not in supported_weight_modes:
        raise ValueError(
            "HLSDAG003: DAG inference supports embedded/full-import runtime weights; "
            f"got {normalized_weights_mode!r}. Tiled external-weight execution is not yet implemented by the DAG emitter."
        )
    if len(getattr(graph, "inputs", []) or []) < 1 or len(getattr(graph, "outputs", []) or []) != 1:
        raise ValueError("HLSDAG004: maintained DAG top requires at least one graph input and exactly one graph output")

    allocation = dict(buffer_allocation or build_hls_buffer_allocation(
        graph, raw_cfg=raw_cfg, tensor_liveness=tensor_liveness
    ))
    tensor_to_buffer = dict(allocation.get("tensor_to_buffer", {}))
    tensor_types, accumulator_type = _resolved_tensor_types(graph, raw_cfg)
    io_names = [str(x) for x in (getattr(graph, "inputs", []) or [])] + [str(x) for x in (getattr(graph, "outputs", []) or [])]
    io_bits = [_cpp_type_bits(tensor_types[name], default=16) for name in io_names if name in tensor_types]
    axis_data_bits = 64 if any(bits > 32 for bits in io_bits) else 32
    defaults = _default_precision(dict(raw_cfg or {}))
    act_spec = defaults["activation"]
    act_bits = _spec_bits(act_spec, default=16)
    if act_bits <= 0 or act_bits > 32 or 32 % act_bits != 0:
        raise ValueError(f"HLSDAG005: activation width {act_bits} is not supported by 32-bit AXIS packing")
    act_per_axis = 32 // act_bits
    input_movement = _io_movement_kind(raw_cfg, "input")
    output_movement = _io_movement_kind(raw_cfg, "output")

    lines: list[str] = [
        "#include <hls_stream.h>",
        "#include <ap_axi_sdata.h>",
        "#include <ap_int.h>",
        '#include "fpgai_types.h"',
        '#include "fpgai_params.h"',
        '#include "layers/activations.h"',
        '#include "layers/dense.h"',
        '#include "layers/conv.h"',
        '#include "layers/pool.h"',
        '#include "layers/attention.h"',
        '#include "layers/tensor.h"',
        "",
        f"typedef ap_axis<{axis_data_bits}, 0, 0, 0> axis_t;",
        "using namespace fpgai;",
        "",
    ]
    if external_composition_plan is not None:
        from fpgai.implementations.hls_composition import package_declarations
        lines.extend(package_declarations(external_composition_plan))
        lines.append("")

    lines.extend([
        "template<typename T>",
        "static inline T bits_to_value(unsigned int bits) {",
        "    union { unsigned int i; float f; } converter;",
        "    converter.i = bits;",
        "    return (T)converter.f;",
        "}",
        "",
        "template<typename T>",
        "static inline unsigned int value_to_bits(T value) {",
        "    union { unsigned int i; float f; } converter;",
        "    converter.f = (float)value;",
        "    return converter.i;",
        "}",
        "",
        "template<typename T, int VALUE_BITS>",
        "static inline T fpgai_unpack_axis_value(axis_t packet, int lane) {",
        "#pragma HLS INLINE",
        "    ap_uint<VALUE_BITS> raw = packet.data.range(((lane + 1) * VALUE_BITS) - 1, lane * VALUE_BITS);",
        "    T value;",
        "    value.range(VALUE_BITS - 1, 0) = raw;",
        "    return value;",
        "}",
        "",
        "template<typename T, int VALUE_BITS>",
        "static inline void fpgai_pack_axis_value(axis_t& packet, T value, int lane) {",
        "#pragma HLS INLINE",
        "    ap_uint<VALUE_BITS> raw = value.range(VALUE_BITS - 1, 0);",
        "    packet.data.range(((lane + 1) * VALUE_BITS) - 1, lane * VALUE_BITS) = raw;",
        "}",
        "",
        f'extern "C" void {top_name}(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {{',
        "#pragma HLS INTERFACE axis port=in_stream",
        "#pragma HLS INTERFACE axis port=out_stream",
        _control_interface_pragma(raw_cfg),
        "",
    ])

    network_execution_mode = requested_network_execution_mode(raw_cfg)
    if network_execution_mode == "dataflow":
        if external_composition_plan is not None:
            raise ValueError("HLSDAG051: network dataflow with external composition is not yet supported")
        lines.extend([
            "#pragma HLS DATAFLOW",
            "    // FPGAI_NETWORK_EXECUTION mode=dataflow physical=pragma",
            "",
        ])
    elif network_execution_mode in {"phase_shared", "parallel"}:
        lines.append(f"    // FPGAI_NETWORK_EXECUTION mode={network_execution_mode} physical=not_emitted")
        lines.append("")
    else:
        lines.append("    // FPGAI_NETWORK_EXECUTION mode=sequential physical=implemented")
        lines.append("")

    lines.extend([
        f"    static const int FPGAI_ACT_BITS = {act_bits};",
        f"    static const int FPGAI_ACT_PER_AXIS = {max(1, axis_data_bits // act_bits)};",
        "",
    ])

    tied_parameter_bindings, tied_parameter_declarations = _tied_parameter_bindings(graph)
    for declaration in tied_parameter_declarations:
        values = declaration["values"]
        lines.append(f"    static const {accumulator_type} {declaration['symbol']}[{values.size}] = {{ {', '.join(f'{float(v):.17g}' for v in values)} }};")
        lines.append(f"    // FPGAI_TIED_PARAMETER owner={declaration['tensor']} physical_symbol={declaration['symbol']}")
    if tied_parameter_declarations:
        lines.append("")

    persistent_slot_by_tensor: dict[str, Mapping[str, Any]] = {}
    external_persistent_slots: list[Mapping[str, Any]] = []
    for slot in allocation.get("slots", []) or []:
        is_persistent = bool(slot.get("persistent", False))
        storage = str(slot.get("storage", "unspecified")).strip().lower()
        external_state = is_persistent and storage in {"ddr", "host", "external"}
        if not external_state:
            qualifier = "static " if is_persistent else ""
            lines.append(f"    {qualifier}{slot['cpp_type']} {slot['name']}[{int(slot['words'])}];")
        else:
            external_persistent_slots.append(dict(slot))
        lines.append(
            f"    // FPGAI_BUFFER_SLOT {slot['name']} tensors={','.join(str(x) for x in slot.get('tensors', []))} persistent={str(is_persistent).lower()}"
        )
        if is_persistent:
            if storage in {"bram", "block_ram"}:
                lines.append(f"#pragma HLS BIND_STORAGE variable={slot['name']} type=ram_2p impl=bram")
            elif storage in {"uram", "ultra_ram"}:
                lines.append(f"#pragma HLS BIND_STORAGE variable={slot['name']} type=ram_2p impl=uram")
            elif storage in {"ddr", "host", "external"}:
                lines.append(f"    // FPGAI_EXTERNAL_STATE_PORT {slot['name']} storage={storage}")
            cursor_name = f"{slot['name']}_cursor"
            cursor_ref = f"{cursor_name}[0]" if external_state else cursor_name
            if not external_state:
                lines.append(f"    static int {cursor_name} = 0;")
            else:
                lines.append(f"    // External state cursor is persisted in {cursor_name}[0] so host reset/import and decode position remain synchronized.")
            for tensor_name in slot.get("tensors", []) or []:
                persistent_slot_by_tensor[str(tensor_name)] = dict(slot, cursor_name=cursor_ref, external=external_state)
    lines.append("")

    for graph_input in getattr(graph, "inputs", []) or []:
        input_name = str(graph_input)
        input_buffer = tensor_to_buffer[input_name]
        input_type = tensor_types[input_name]
        input_words = _tensor_words(graph, input_name)
        input_bits = _cpp_type_bits(input_type, default=act_bits)
        if input_bits <= 0 or input_bits > axis_data_bits or axis_data_bits % input_bits != 0:
            raise ValueError(f"HLSDAG059: input tensor {input_name!r} scalar width {input_bits} is not supported by {axis_data_bits}-bit AXIS packing")
        input_per_axis = axis_data_bits // input_bits
        lines.extend([
            f"    // FPGAI_INPUT_SEGMENT tensor={input_name} words={input_words} bits={input_bits}",
            f"    // FPGAI_BUFFER_PROVENANCE buffer={input_buffer} tensor={input_name}",
        ])
        if input_movement.startswith("m_axi"):
            if input_bits > 32:
                raise ValueError(f"HLSDAG091: m_axi input currently supports scalar widths up to 32 bits, got {input_bits} for {input_name!r}")
            if input_movement == "m_axi_tiled":
                tile = _io_tile_size(raw_cfg, "input", input_words)
                lines.extend([
                    f"    static const int FPGAI_INPUT_TILE_SIZE = {tile};",
                    f"    {input_type} input_tile[FPGAI_INPUT_TILE_SIZE];",
                    "#pragma HLS BIND_STORAGE variable=input_tile type=ram_1p impl=bram",
                    f"    // m_axi tiled input import: input_mem -> input_tile -> {input_buffer}.",
                    f"    for (int tile_base = 0; tile_base < {input_words}; tile_base += FPGAI_INPUT_TILE_SIZE) {{",
                    f"        int tile_count = ((tile_base + FPGAI_INPUT_TILE_SIZE) <= {input_words}) ? FPGAI_INPUT_TILE_SIZE : ({input_words} - tile_base);",
                    "        for (int lane = 0; lane < FPGAI_INPUT_TILE_SIZE; ++lane) {",
                    "#pragma HLS PIPELINE II=1",
                    "            if (lane < tile_count) {",
                    f"                ap_uint<{input_bits}> raw = input_mem[tile_base + lane].range({input_bits - 1}, 0);",
                    f"                {input_type} value; value.range({input_bits - 1}, 0) = raw;",
                    "                input_tile[lane] = value;",
                    "            }",
                    "        }",
                    "        for (int lane = 0; lane < FPGAI_INPUT_TILE_SIZE; ++lane) {",
                    "#pragma HLS PIPELINE II=1",
                    f"            if (lane < tile_count) {input_buffer}[tile_base + lane] = input_tile[lane];",
                    "        }",
                    "    }",
                ])
            else:
                lines.extend([
                    f"    // m_axi full input import: input_mem -> {input_buffer}.",
                    f"    for (int index = 0; index < {input_words}; ++index) {{",
                    "#pragma HLS PIPELINE II=1",
                    f"        ap_uint<{input_bits}> raw = input_mem[index].range({input_bits - 1}, 0);",
                    f"        {input_type} value; value.range({input_bits - 1}, 0) = raw;",
                    f"        {input_buffer}[index] = value;",
                    "    }",
                ])
        elif input_movement == "axi_stream_tiled":
            tile = _io_tile_size(raw_cfg, "input", input_words)
            lines.extend([
                f"    static const int FPGAI_AXIS_INPUT_TILE_SIZE = {tile};",
                f"    {input_type} input_tile[FPGAI_AXIS_INPUT_TILE_SIZE];",
                "#pragma HLS BIND_STORAGE variable=input_tile type=ram_1p impl=bram",
                f"    for (int tile_base = 0; tile_base < {input_words}; tile_base += FPGAI_AXIS_INPUT_TILE_SIZE) {{",
                f"        int tile_count = ((tile_base + FPGAI_AXIS_INPUT_TILE_SIZE) <= {input_words}) ? FPGAI_AXIS_INPUT_TILE_SIZE : ({input_words} - tile_base);",
                "        for (int lane = 0; lane < FPGAI_AXIS_INPUT_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                "            if (lane < tile_count) {",
                "                axis_t packet = in_stream.read();",
                f"                input_tile[lane] = fpgai_unpack_axis_value<{input_type}, {input_bits}>(packet, 0);",
                "            }",
                "        }",
                "        for (int lane = 0; lane < FPGAI_AXIS_INPUT_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                f"            if (lane < tile_count) {input_buffer}[tile_base + lane] = input_tile[lane];",
                "        }",
                "    }",
            ])
        else:
            lines.extend([
                f"    for (int base = 0; base < {input_words}; base += {input_per_axis}) {{",
                "#pragma HLS PIPELINE II=1",
                "        axis_t packet = in_stream.read();",
                f"        for (int lane = 0; lane < {input_per_axis}; ++lane) {{",
                "#pragma HLS UNROLL",
                "            int index = base + lane;",
                f"            if (index < {input_words}) {input_buffer}[index] = fpgai_unpack_axis_value<{input_type}, {'FPGAI_ACT_BITS' if input_bits == act_bits else input_bits}>(packet, lane);",
                "        }",
                "    }",
            ])
        lines.append("")

    parameter_index = 0
    for index, op in enumerate(graph.ops):
        runtime_inputs = [str(x) for x in (getattr(op, "inputs", []) or []) if str(x) not in getattr(graph, "constants", {})]
        outputs = [str(x) for x in (getattr(op, "outputs", []) or [])]
        binding = external_composition_plan.binding_for_node(op.name) if external_composition_plan is not None else None
        if binding is not None:
            from fpgai.implementations.hls_integration import parse_hls_abi, HLSFlatArrayABI
            abi = parse_hls_abi(binding.contract)
            lines.append(f"    // DAG node {index}: {op.op_type} ({op.name})")
            for name in outputs:
                lines.append(f"    // FPGAI_BUFFER_PROVENANCE buffer={tensor_to_buffer[name]} tensor={name}")
            if isinstance(abi, HLSFlatArrayABI):
                if len(runtime_inputs) != 1 or len(outputs) != 1:
                    raise RuntimeError(f"HLSDAG007: external flat_array_v1 node {op.name!r} requires one runtime tensor input/output")
                output_name = outputs[0]; output_buffer = tensor_to_buffer[output_name]; output_type = tensor_types[output_name]; output_words = _tensor_words(graph, output_name)
                input_name_for_op = runtime_inputs[0]
                from fpgai.implementations.hls_composition import emit_external_call
                lines.extend(emit_external_call(binding, current_buffer=tensor_to_buffer[input_name_for_op], current_type=tensor_types[input_name_for_op], output_buffer=output_buffer, output_type=output_type))
            else:
                from fpgai.implementations.hls_composition import emit_external_tensor_ports_call
                lines.extend(emit_external_tensor_ports_call(
                    binding,
                    input_buffers={name: tensor_to_buffer[name] for name in runtime_inputs},
                    input_types={name: tensor_types[name] for name in runtime_inputs},
                    output_buffers={name: tensor_to_buffer[name] for name in outputs},
                    output_types={name: tensor_types[name] for name in outputs},
                ))
            lines.append("")
            continue

        if len(outputs) != 1:
            raise RuntimeError(f"HLSDAG006: node {op.name!r} must have exactly one output in the current built-in DAG profile")
        output_name = outputs[0]
        output_buffer = tensor_to_buffer[output_name]
        output_type = tensor_types[output_name]
        output_words = _tensor_words(graph, output_name)
        lines.append(f"    // DAG node {index}: {op.op_type} ({op.name})")
        lines.append(f"    // FPGAI_BUFFER_PROVENANCE buffer={output_buffer} tensor={output_name}")

        if op.op_type in {"Relu", "Sigmoid", "LeakyRelu", "Identity", "Cast", "Squeeze", "Unsqueeze", "Flatten", "Reshape"}:
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG008: unary node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            source_buffer = tensor_to_buffer[source_name]
            source_type = tensor_types[source_name]
            if op.op_type == "Relu":
                qcfg = (getattr(op, "attrs", {}) or {}).get("quantized_relu")
                if isinstance(qcfg, Mapping):
                    lines.append(
                        f"    relu_quantized<{output_words}, {source_type}, {output_type}>("
                        f"{source_buffer}, {output_buffer}, {int(qcfg.get('input_zero', 0))}, "
                        f"{int(qcfg.get('multiplier', 1))}, {int(qcfg.get('shift', 0))}, "
                        f"{int(qcfg.get('output_zero', 0))}, {int(qcfg.get('qmin', -128))}, {int(qcfg.get('qmax', 127))}, "
                        f"{int(qcfg.get('rounding_mode', 0))}, {int(qcfg.get('saturation_mode', 0))});"
                    )
                else:
                    lines.append(f"    relu_typed<{output_words}, {source_type}, {output_type}>({source_buffer}, {output_buffer});")
            elif op.op_type == "Sigmoid":
                lines.append(f"    sigmoid_typed<{output_words}, {source_type}, {output_type}, {accumulator_type}>({source_buffer}, {output_buffer});")
            elif op.op_type == "LeakyRelu":
                alpha = float((getattr(op, "attrs", {}) or {}).get("alpha", 0.1))
                lines.append(f"    leaky_relu_typed<{output_words}, {source_type}, {output_type}, {accumulator_type}>({source_buffer}, {output_buffer}, ({accumulator_type}){alpha:.17g});")
            else:
                source_shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
                spatial_shape = source_shape
                if len(spatial_shape) == 4 and spatial_shape[0] == 1:
                    spatial_shape = spatial_shape[1:]
                output_shape = tuple(int(x) for x in graph.get_tensor(output_name).shape)
                # Conv/pool HLS kernels use HWC-flat internal feature-map storage while
                # ONNX Flatten/Reshape after an NCHW feature map requires channel-major
                # flattening. Mirror the established linear-top layout bridge here so
                # branch-aware DAG lowering preserves model semantics before Dense.
                producer = None
                for candidate in graph.ops:
                    if source_name in [str(x) for x in (getattr(candidate, "outputs", []) or [])]:
                        producer = candidate
                        break
                spatial_hwc_producers = {
                    "Conv", "MaxPool", "AvgPool", "Resize",
                }
                layout_preserving = {
                    "Relu", "Sigmoid", "LeakyRelu", "Identity", "Cast",
                    "BatchNormalization",
                }
                producer_type = None if producer is None else str(producer.op_type)
                if producer_type in layout_preserving and producer is not None:
                    # One hop is sufficient for the maintained CNN path (pool -> optional
                    # activation -> flatten) and avoids guessing layout for arbitrary rank-3
                    # sequence tensors.
                    upstream = [str(x) for x in (getattr(producer, "inputs", []) or []) if str(x) not in getattr(graph, "constants", {})]
                    if upstream:
                        upstream_name = upstream[0]
                        for candidate in graph.ops:
                            if upstream_name in [str(x) for x in (getattr(candidate, "outputs", []) or [])]:
                                if str(candidate.op_type) in spatial_hwc_producers:
                                    producer_type = str(candidate.op_type)
                                break
                needs_spatial_flatten_bridge = (
                    op.op_type in {"Flatten", "Reshape"}
                    and len(spatial_shape) == 3
                    and producer_type in spatial_hwc_producers
                    and len(output_shape) <= 2
                    and output_words == spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
                )
                if needs_spatial_flatten_bridge:
                    channels, height, width = spatial_shape
                    lines.extend([
                        f"    // FPGAI layout bridge: internal HWC-flat -> ONNX NCHW flatten for {op.op_type}",
                        f"    for (int channel = 0; channel < {channels}; ++channel) {{",
                        f"        for (int row = 0; row < {height}; ++row) {{",
                        f"            for (int column = 0; column < {width}; ++column) {{",
                        f"                const int source = (row * {width} + column) * {channels} + channel;",
                        f"                const int destination = (channel * {height} + row) * {width} + column;",
                        f"                {output_buffer}[destination] = ({output_type}){source_buffer}[source];",
                        "            }",
                        "        }",
                        "    }",
                    ])
                else:
                    lines.append(f"    reshape_copy_typed<{output_words}, {source_type}, {output_type}>({source_buffer}, {output_buffer});")
        elif op.op_type == "Dense":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG012: Dense node {op.name!r} requires one runtime activation input")
            source_name = runtime_inputs[0]
            source_words = _tensor_words(graph, source_name)
            lines.append(
                f"    dense_out_in<{source_words}, {output_words}, {tensor_types[source_name]}, {output_type}, op{index}_wgt_t, op{index}_bias_t, op{index}_acc_t>("
                f"{tensor_to_buffer[source_name]}, {output_buffer}, W{parameter_index}, B{parameter_index});"
            )
            parameter_index += 1
        elif op.op_type == "Conv":
            if len(runtime_inputs) != 1 or len(getattr(op, "inputs", []) or []) < 2:
                raise RuntimeError(f"HLSDAG013: Conv node {op.name!r} requires one runtime input plus embedded weights")
            source_name = runtime_inputs[0]
            in_spec = graph.get_tensor(source_name); out_spec = graph.get_tensor(output_name)
            in_shape = tuple(int(x) for x in in_spec.shape); out_shape = tuple(int(x) for x in out_spec.shape)
            if len(in_shape) == 4 and in_shape[0] == 1: in_shape = in_shape[1:]
            if len(out_shape) == 4 and out_shape[0] == 1: out_shape = out_shape[1:]
            if len(in_shape) != 3 or len(out_shape) != 3:
                raise RuntimeError(f"HLSDAG014: Conv node {op.name!r} requires NCHW/CHW static shapes")
            ic, ih, iw = in_shape; oc, oh, ow = out_shape
            weight_name = op.inputs[1]
            weight = (getattr(graph, "constants", {}) or {}).get(weight_name)
            weight_shape = tuple(int(x) for x in getattr(weight, "shape", ()))
            if len(weight_shape) != 4 or weight_shape[2] != weight_shape[3]:
                raise RuntimeError(f"HLSDAG015: Conv node {op.name!r} requires square embedded kernels")
            kernel = weight_shape[2]; strides = (getattr(op, "attrs", {}) or {}).get("strides", [1,1]); pads=(getattr(op, "attrs", {}) or {}).get("pads", [0,0,0,0])
            groups = int((getattr(op, "attrs", {}) or {}).get("group", (getattr(op, "attrs", {}) or {}).get("groups", 1)))
            if groups <= 0 or ic % groups != 0 or oc % groups != 0:
                raise RuntimeError(f"HLSDAG016: Conv node {op.name!r} has invalid group count {groups} for IC={ic}, OC={oc}")
            if weight_shape[1] != ic // groups:
                raise RuntimeError(f"HLSDAG016: Conv node {op.name!r} weight input-channel extent must equal IC/groups ({ic // groups})")
            if int(strides[0]) != int(strides[1]) or not (int(pads[0]) == int(pads[1]) == int(pads[2]) == int(pads[3])):
                raise RuntimeError(f"HLSDAG016: Conv node {op.name!r} requires symmetric stride/padding")
            qcfg = (getattr(op, "attrs", {}) or {}).get("quantized_conv")
            if groups != 1 and isinstance(qcfg, Mapping):
                raise RuntimeError(f"HLSDAG017: quantized grouped Conv node {op.name!r} is not yet implemented")
            if isinstance(qcfg, Mapping):
                weight_zero = [int(v) for v in qcfg.get("weight_zero", [])]
                multipliers = [int(v) for v in qcfg.get("multipliers", [])]
                shifts_q = [int(v) for v in qcfg.get("shifts", [])]
                if len(weight_zero) != oc or len(multipliers) != oc or len(shifts_q) != oc:
                    raise RuntimeError(f"HLSDAG017: quantized Conv node {op.name!r} channel parameter count mismatch")
                wz_name = f"fpgai_q_wzero_{index}"
                mul_name = f"fpgai_q_mult_{index}"
                sh_name = f"fpgai_q_shift_{index}"
                lines.append(f"    static const int {wz_name}[{oc}] = {{ {', '.join(str(v) for v in weight_zero)} }};")
                lines.append(f"    static const int {mul_name}[{oc}] = {{ {', '.join(str(v) for v in multipliers)} }};")
                lines.append(f"    static const int {sh_name}[{oc}] = {{ {', '.join(str(v) for v in shifts_q)} }};")
                lines.append(
                    f"    conv2d_quantized<{ih}, {iw}, {ic}, {oh}, {ow}, {oc}, {kernel}, {int(strides[0])}, {int(pads[0])}, {tensor_types[source_name]}, {output_type}, op{index}_wgt_t, op{index}_bias_t, op{index}_acc_t>("
                    f"{tensor_to_buffer[source_name]}, {output_buffer}, "
                    f"reinterpret_cast<const op{index}_wgt_t*>(W{parameter_index}), B{parameter_index}, "
                    f"{int(qcfg.get('input_zero', 0))}, {wz_name}, {int(qcfg.get('output_zero', 0))}, "
                    f"{mul_name}, {sh_name}, {int(qcfg.get('qmin', -128))}, {int(qcfg.get('qmax', 127))}, "
                    f"{int(qcfg.get('rounding_mode', 0))}, {int(qcfg.get('saturation_mode', 0))});"
                )
            else:
                kernel_name = "conv2d" if groups == 1 else "conv2d_grouped"
                group_arg = "" if groups == 1 else f", {groups}"
                lines.append(
                    f"    {kernel_name}<{ih}, {iw}, {ic}, {oh}, {ow}, {oc}, {kernel}, {int(strides[0])}, {int(pads[0])}{group_arg}, {tensor_types[source_name]}, {output_type}, op{index}_wgt_t, op{index}_bias_t, op{index}_acc_t>("
                    f"{tensor_to_buffer[source_name]}, {output_buffer}, "
                    f"reinterpret_cast<const op{index}_wgt_t*>(W{parameter_index}), B{parameter_index});"
                )
            parameter_index += 1
        elif op.op_type in {"MaxPool", "AvgPool"}:
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG093: {op.op_type} node {op.name!r} requires exactly one runtime tensor input")
            source_name = runtime_inputs[0]
            in_shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            out_shape = tuple(int(x) for x in graph.get_tensor(output_name).shape)
            if len(in_shape) == 4 and in_shape[0] == 1:
                in_shape = in_shape[1:]
            if len(out_shape) == 4 and out_shape[0] == 1:
                out_shape = out_shape[1:]
            if len(in_shape) != 3 or len(out_shape) != 3:
                raise RuntimeError(f"HLSDAG094: {op.op_type} node {op.name!r} requires static NCHW/CHW tensors")
            ic, ih, iw = in_shape
            oc, oh, ow = out_shape
            if oc != ic:
                raise RuntimeError(f"HLSDAG094: {op.op_type} node {op.name!r} must preserve channel count")
            attrs = getattr(op, "attrs", {}) or {}
            kernel = attrs.get("kernel_shape", [2, 2])
            strides = attrs.get("strides", [2, 2])
            pads = attrs.get("pads", [0, 0, 0, 0])
            dilations = attrs.get("dilations", [1, 1])
            ceil_mode = int(attrs.get("ceil_mode", 0) or 0)
            if len(kernel) != 2 or int(kernel[0]) != int(kernel[1]):
                raise RuntimeError(f"HLSDAG095: {op.op_type} node {op.name!r} requires a square kernel")
            if len(strides) != 2 or int(strides[0]) != int(strides[1]):
                raise RuntimeError(f"HLSDAG095: {op.op_type} node {op.name!r} requires equal spatial strides")
            if any(int(v) != 0 for v in pads):
                raise RuntimeError(f"HLSDAG095: {op.op_type} node {op.name!r} currently requires zero padding")
            if any(int(v) != 1 for v in dilations) or ceil_mode != 0:
                raise RuntimeError(f"HLSDAG095: {op.op_type} node {op.name!r} currently requires dilation=1 and ceil_mode=0")
            fn = "maxpool2d_typed" if op.op_type == "MaxPool" else "avgpool2d_typed"
            if op.op_type == "MaxPool":
                templates = f"{tensor_types[source_name]}, {output_type}"
            else:
                templates = f"{tensor_types[source_name]}, {output_type}, {accumulator_type}"
            lines.append(
                f"    {fn}<{ih}, {iw}, {ic}, {int(kernel[0])}, {int(strides[0])}, {oh}, {ow}, {templates}>("
                f"{tensor_to_buffer[source_name]}, {output_buffer});"
            )
        elif op.op_type == "Transpose":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG018: Transpose node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            source_shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            perm = tuple(int(x) for x in (getattr(op, "attrs", {}) or {}).get("perm", ()))
            if len(source_shape) == 3 and source_shape[0] == 1 and perm in {(0, 2, 1), ()}:
                rows, cols = source_shape[1], source_shape[2]
            elif len(source_shape) == 2 and perm in {(1, 0), ()}:
                rows, cols = source_shape
            else:
                raise RuntimeError(f"HLSDAG019: attention Transpose node {op.name!r} requires rank-2 or batch-1 rank-3 last-two-dimension transpose")
            lines.append(
                f"    transpose_2d<{rows}, {cols}, {tensor_types[source_name]}, {output_type}>("
                f"{tensor_to_buffer[source_name]}, {output_buffer});"
            )
        elif op.op_type == "MatMul":
            op_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            if len(op_inputs) != 2:
                raise RuntimeError(f"HLSDAG020: MatMul node {op.name!r} requires exactly two tensor inputs")
            left_name, right_name = op_inputs
            left_shape = tuple(int(x) for x in graph.get_tensor(left_name).shape)
            right_shape = tuple(int(x) for x in graph.get_tensor(right_name).shape)
            if len(left_shape) == 3 and left_shape[0] == 1: left_shape = left_shape[1:]
            if len(right_shape) == 3 and right_shape[0] == 1: right_shape = right_shape[1:]
            if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[1] != right_shape[0]:
                raise RuntimeError(f"HLSDAG021: MatMul node {op.name!r} currently requires static rank-2 or batch-1 rank-3 compatible matrices")
            m, k = left_shape; _, n = right_shape
            schedule = getattr(getattr(op, "semantics", None), "schedule", {}) or {}
            tile_m = max(1, min(int(schedule.get("tile_m", 1)), m))
            tile_n = max(1, min(int(schedule.get("tile_n", 1)), n))
            tile_k = max(1, min(int(schedule.get("tile_k", 1)), k))
            def _matmul_arg(name: str, role: str):
                if name in (getattr(graph, "constants", {}) or {}):
                    tied = tied_parameter_bindings.get(name)
                    if tied is not None:
                        return tied["symbol"], accumulator_type, tied.get("view", "native")
                    import numpy as np
                    arr = np.asarray(graph.constants[name], dtype=float).reshape(-1)
                    expected = 1
                    for dim in tuple(int(x) for x in graph.get_tensor(name).shape): expected *= dim
                    if arr.size != expected:
                        raise RuntimeError(f"HLSDAG045: MatMul {role} constant {name!r} size does not match its tensor shape")
                    symbol = f"fpgai_matmul_{role}_{index}"
                    lines.append(f"    static const {accumulator_type} {symbol}[{arr.size}] = {{ {', '.join(f'{float(v):.17g}' for v in arr)} }};")
                    return symbol, accumulator_type, "native"
                if name not in tensor_to_buffer:
                    raise RuntimeError(f"HLSDAG046: MatMul {role} tensor {name!r} has no runtime buffer")
                return tensor_to_buffer[name], tensor_types[name], "native"
            left_arg, left_type, left_view = _matmul_arg(left_name, "left")
            right_arg, right_type, right_view = _matmul_arg(right_name, "right")
            if left_view != "native":
                raise RuntimeError(f"HLSDAG100: transposed tied parameters are currently supported only as the right MatMul operand")
            kernel = "matmul_tiled_right_transposed" if right_view == "transpose" else "matmul_tiled"
            lines.append(
                f"    {kernel}<{m}, {k}, {n}, {left_type}, {right_type}, {output_type}, {accumulator_type}, {tile_m}, {tile_n}, {tile_k}>("
                f"{left_arg}, {right_arg}, {output_buffer});"
            )
        elif op.op_type == "Mul":
            if len(runtime_inputs) == 1 and len(getattr(op, "inputs", []) or []) == 2:
                source_name = runtime_inputs[0]
                constant_name = next(str(x) for x in op.inputs if str(x) != source_name)
                constant = (getattr(graph, "constants", {}) or {}).get(constant_name)
                import numpy as np
                arr = np.asarray(constant, dtype=float).reshape(-1)
                if arr.size == 1:
                    scale_value = float(arr[0])
                    lines.append(
                        f"    scale_vector<{output_words}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                        f"{tensor_to_buffer[source_name]}, {output_buffer}, ({accumulator_type}){scale_value:.17g});"
                    )
                else:
                    import numpy as np
                    source_shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
                    raw_const = np.asarray(constant, dtype=float)
                    if len(source_shape) >= 1 and raw_const.size == source_shape[-1]:
                        cols = source_shape[-1]; rows = output_words // cols
                        symbol = f"fpgai_mul_scale_{index}"
                        flat = raw_const.reshape(-1)
                        lines.append(f"    static const {accumulator_type} {symbol}[{cols}] = {{ {', '.join(f'{float(v):.17g}' for v in flat)} }};")
                        lines.append(
                            f"    mul_rows_by_col_vector<{rows}, {cols}, {tensor_types[source_name]}, {accumulator_type}, {output_type}, {accumulator_type}>("
                            f"{tensor_to_buffer[source_name]}, {symbol}, {output_buffer});"
                        )
                    else:
                        out_shape = tuple(int(x) for x in graph.get_tensor(output_name).shape)
                        try:
                            expanded = np.broadcast_to(raw_const, out_shape).reshape(-1)
                        except ValueError as exc:
                            raise RuntimeError(f"HLSDAG022: Mul node {op.name!r} constant is not broadcast-compatible with output shape") from exc
                        if _tensor_words(graph, source_name) != output_words:
                            raise RuntimeError(f"HLSDAG022: Mul constant-broadcast path requires runtime input to match output size")
                        symbol = f"fpgai_mul_const_{index}"
                        lines.append(f"    static const {accumulator_type} {symbol}[{output_words}] = {{ {', '.join(f'{float(v):.17g}' for v in expanded)} }};")
                        lines.append(
                            f"    mul_vectors<{output_words}, {tensor_types[source_name]}, {accumulator_type}, {output_type}, {accumulator_type}>("
                            f"{tensor_to_buffer[source_name]}, {symbol}, {output_buffer});"
                        )
            elif len(runtime_inputs) == 2:
                left_name, right_name = runtime_inputs
                if _tensor_words(graph, left_name) != output_words or _tensor_words(graph, right_name) != output_words:
                    raise RuntimeError(f"HLSDAG023: Mul node {op.name!r} tensor lowering requires equal flattened input/output sizes")
                lines.append(
                    f"    mul_vectors<{output_words}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer});"
                )
            else:
                raise RuntimeError(f"HLSDAG023: Mul node {op.name!r} requires two equal-shaped runtime tensors or one runtime tensor plus scalar constant")
        elif op.op_type == "SiLU":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG047: SiLU node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            if _tensor_words(graph, source_name) != output_words:
                raise RuntimeError(f"HLSDAG048: SiLU node {op.name!r} requires equal flattened input/output sizes")
            lines.append(
                f"    silu_vector<{output_words}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                f"{tensor_to_buffer[source_name]}, {output_buffer});"
            )
        elif op.op_type == "Concat":
            names = [str(x) for x in op.inputs]
            if len(names) < 2 or any(name not in tensor_to_buffer for name in names):
                raise RuntimeError(f"HLSDAG049: Concat node {op.name!r} requires at least two runtime tensor inputs")
            first_shape = tuple(int(x) for x in graph.get_tensor(names[0]).shape)
            axis = normalize_axis(int((getattr(op, "attrs", {}) or {}).get("axis", 0)), len(first_shape))
            geometries = []
            for name in names:
                shape = tuple(int(x) for x in graph.get_tensor(name).shape)
                outer_i, axis_i, inner_i = axis_geometry(shape, axis)
                geometries.append((outer_i, axis_i, inner_i))
            outer, _, inner = geometries[0]
            if any(outer_i != outer or inner_i != inner for outer_i, _, inner_i in geometries[1:]):
                raise RuntimeError(f"HLSDAG051: Concat node {op.name!r} non-axis dimensions must match")
            out_axis = sum(axis_i for _, axis_i, _ in geometries)
            if len(names) == 2:
                left_name, right_name = names
                left_axis = geometries[0][1]; right_axis = geometries[1][1]
                lines.append(
                    f"    concat_axis<{outer}, {left_axis}, {right_axis}, {inner}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}>"
                    f"({tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer});"
                )
            else:
                offset = 0
                for name, (_, input_axis, _) in zip(names, geometries):
                    lines.append(
                        f"    concat_axis_segment<{outer}, {out_axis}, {input_axis}, {offset}, {inner}, {tensor_types[name]}, {output_type}>"
                        f"({tensor_to_buffer[name]}, {output_buffer});"
                    )
                    offset += input_axis

        elif op.op_type == "Slice":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG052: Slice node {op.name!r} requires one runtime data tensor and static slice parameters")
            source_name = runtime_inputs[0]; shape = tuple(int(x) for x in graph.get_tensor(source_name).shape); spec = resolve_slice_spec(graph, op, shape)
            outer, in_axis, inner = axis_geometry(shape, spec["axis"]); out_axis = spec["length"]
            lines.append(
                f"    slice_axis<{outer}, {in_axis}, {spec['start']}, {out_axis}, {inner}, {tensor_types[source_name]}, {output_type}>"
                f"({tensor_to_buffer[source_name]}, {output_buffer});"
            )

        elif op.op_type == "Resize":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG053: Resize node {op.name!r} requires one runtime data tensor and static sizes/scales")
            source_name = runtime_inputs[0]; shape = tuple(int(x) for x in graph.get_tensor(source_name).shape); out_shape = resolve_resize_shape(graph, op, shape)
            attrs = getattr(op, "attrs", {}) or {}; mode = str(attrs.get("mode", "nearest")).lower()
            if len(shape) != 4 or len(out_shape) != 4 or shape[:2] != out_shape[:2] or mode != "nearest":
                raise RuntimeError(f"HLSDAG054: Resize node {op.name!r} current HLS implementation requires static NCHW nearest resize with unchanged N/C")
            coord_code, nearest_code = _resize_mode_codes(attrs)
            b, c, ih, iw = shape; _, _, oh, ow = out_shape
            lines.append(
                f"    resize_nearest_nchw<{b}, {c}, {ih}, {iw}, {oh}, {ow}, {coord_code}, {nearest_code}, {tensor_types[source_name]}, {output_type}>"
                f"({tensor_to_buffer[source_name]}, {output_buffer});"
            )

        elif op.op_type == "Gather":
            names = [str(x) for x in op.inputs]
            if len(names) < 2:
                raise RuntimeError(f"HLSDAG055: Gather node {op.name!r} requires data and indices")
            data_name, indices_name = names[:2]; data_shape = tuple(int(x) for x in graph.get_tensor(data_name).shape); index_shape = tuple(int(x) for x in graph.get_tensor(indices_name).shape)
            axis = normalize_axis(int((getattr(op, "attrs", {}) or {}).get("axis", 0)), len(data_shape))
            if axis != 0 or len(data_shape) != 2:
                raise RuntimeError(f"HLSDAG056: Gather node {op.name!r} current HLS implementation requires rank-2 data with axis=0")
            rows, width = data_shape; index_count = _tensor_words(graph, indices_name)
            if data_name in (getattr(graph, "constants", {}) or {}):
                tied = tied_parameter_bindings.get(data_name)
                if tied is not None:
                    if tied.get("view") != "native":
                        raise RuntimeError(f"HLSDAG099: Gather requires native view for tied parameter {data_name!r}")
                    data_arg, data_type = tied["symbol"], accumulator_type
                else:
                    import numpy as np
                    arr = np.asarray(graph.constants[data_name], dtype=float).reshape(-1)
                    symbol = f"fpgai_gather_data_{index}"
                    lines.append(f"    static const {accumulator_type} {symbol}[{arr.size}] = {{ {', '.join(f'{float(v):.17g}' for v in arr)} }};")
                    data_arg, data_type = symbol, accumulator_type
            else:
                if data_name not in tensor_to_buffer:
                    raise RuntimeError(f"HLSDAG057: Gather data tensor {data_name!r} has no runtime buffer")
                data_arg, data_type = tensor_to_buffer[data_name], tensor_types[data_name]
            if indices_name not in tensor_to_buffer:
                raise RuntimeError(f"HLSDAG058: Gather indices tensor {indices_name!r} must be runtime-accessible")
            lines.append(
                f"    gather_rows<{rows}, {width}, {index_count}, {data_type}, {tensor_types[indices_name]}, {output_type}>"
                f"({data_arg}, {tensor_to_buffer[indices_name]}, {output_buffer});"
            )

        elif op.op_type == "KVCacheUpdate":
            names = [str(x) for x in op.inputs]
            if len(names) < 2:
                raise RuntimeError(f"HLSDAG061: KVCacheUpdate node {op.name!r} requires persistent state and update tensors")
            state_name, update_name = names[:2]
            state_slot = persistent_slot_by_tensor.get(state_name)
            if state_slot is None:
                raise RuntimeError(f"HLSDAG062: KVCacheUpdate state tensor {state_name!r} must carry persistent runtime-state semantics")
            if update_name not in tensor_to_buffer:
                raise RuntimeError(f"HLSDAG063: KVCacheUpdate update tensor {update_name!r} has no runtime buffer")
            state_shape = tuple(int(x) for x in graph.get_tensor(state_name).shape)
            update_shape = tuple(int(x) for x in graph.get_tensor(update_name).shape)
            axis = normalize_axis(int((getattr(op, "attrs", {}) or {}).get("sequence_axis", -2)), len(state_shape))
            if len(state_shape) != len(update_shape):
                raise RuntimeError(f"HLSDAG064: KVCacheUpdate requires state/update tensors with equal rank")
            if any(state_shape[d] != update_shape[d] for d in range(len(state_shape)) if d != axis):
                raise RuntimeError(f"HLSDAG065: KVCacheUpdate non-sequence dimensions must match")
            outer, capacity, inner = axis_geometry(state_shape, axis)
            outer_u, update_axis, inner_u = axis_geometry(update_shape, axis)
            if outer != outer_u or inner != inner_u:
                raise RuntimeError(f"HLSDAG066: KVCacheUpdate flattened axis geometry mismatch")
            declared_capacity = int((getattr(op, "attrs", {}) or {}).get("capacity", capacity))
            if declared_capacity != capacity:
                raise RuntimeError(f"HLSDAG067: KVCacheUpdate capacity attribute must equal persistent state axis extent")
            if str((getattr(op, "attrs", {}) or {}).get("update_policy", "append")).lower() != "append":
                raise RuntimeError(f"HLSDAG068: current persistent HLS state backend implements append policy only")
            state_semantics = getattr(getattr(graph.get_tensor(state_name), "semantics", None), "state", None)
            default_overflow = getattr(state_semantics, "overflow_policy", "saturate") if state_semantics is not None else "saturate"
            overflow_policy = str((getattr(op, "attrs", {}) or {}).get("overflow_policy", default_overflow)).lower().replace("-", "_")
            if overflow_policy != "saturate":
                raise RuntimeError(f"HLSDAG085: current KVCacheUpdate HLS backend supports overflow_policy=saturate only")
            lines.append(
                f"    persistent_state_append_axis<{outer}, {capacity}, {update_axis}, {inner}, {tensor_types[state_name]}, {tensor_types[update_name]}>("
                f"{state_slot['name']}, {tensor_to_buffer[update_name]}, {state_slot['cursor_name']});"
            )
            if output_words == _tensor_words(graph, state_name):
                lines.append(
                    f"    persistent_state_snapshot<{output_words}, {tensor_types[state_name]}, {output_type}>("
                    f"{state_slot['name']}, {output_buffer});"
                )
            elif output_words == _tensor_words(graph, update_name):
                lines.append(
                    f"    copy_vector<{output_words}, {tensor_types[update_name]}, {output_type}>("
                    f"{tensor_to_buffer[update_name]}, {output_buffer});"
                )
            else:
                raise RuntimeError(f"HLSDAG069: KVCacheUpdate output must represent either the updated state or appended update tensor")

        elif op.op_type == "PersistentStateRead":
            names = [str(x) for x in op.inputs]
            if len(names) != 1:
                raise RuntimeError(f"HLSDAG070: PersistentStateRead node {op.name!r} requires exactly one persistent state tensor")
            state_name = names[0]
            state_slot = persistent_slot_by_tensor.get(state_name)
            if state_slot is None:
                raise RuntimeError(f"HLSDAG071: PersistentStateRead tensor {state_name!r} is not persistent HLS state")
            if output_words != _tensor_words(graph, state_name):
                raise RuntimeError(f"HLSDAG072: PersistentStateRead output shape must match persistent state tensor")
            lines.append(
                f"    persistent_state_snapshot<{output_words}, {tensor_types[state_name]}, {output_type}>"
                f"({state_slot['name']}, {output_buffer});"
            )

        elif op.op_type == "PersistentStateLength":
            names = [str(x) for x in op.inputs]
            if len(names) != 1:
                raise RuntimeError(f"HLSDAG073: PersistentStateLength node {op.name!r} requires exactly one persistent state tensor")
            state_name = names[0]
            state_slot = persistent_slot_by_tensor.get(state_name)
            if state_slot is None:
                raise RuntimeError(f"HLSDAG074: PersistentStateLength tensor {state_name!r} is not persistent HLS state")
            if output_words != 1:
                raise RuntimeError(f"HLSDAG075: PersistentStateLength output must be a one-element integer tensor")
            lines.append(
                f"    persistent_state_length<{output_type}>({state_slot['cursor_name']}, {output_buffer});"
            )

        elif op.op_type == "PersistentStateReset":
            names = [str(x) for x in op.inputs]
            if len(names) != 2:
                raise RuntimeError(f"HLSDAG076: PersistentStateReset node {op.name!r} requires state and one-element reset flag tensors")
            state_name, flag_name = names
            state_slot = persistent_slot_by_tensor.get(state_name)
            if state_slot is None:
                raise RuntimeError(f"HLSDAG077: PersistentStateReset tensor {state_name!r} is not persistent HLS state")
            if flag_name not in tensor_to_buffer or _tensor_words(graph, flag_name) != 1:
                raise RuntimeError(f"HLSDAG078: PersistentStateReset reset flag must be a one-element runtime tensor")
            state_words = _tensor_words(graph, state_name)
            lines.append(
                f"    persistent_state_reset_if<{state_words}, {tensor_types[state_name]}, {tensor_types[flag_name]}>"
                f"({state_slot['name']}, {tensor_to_buffer[flag_name]}, {state_slot['cursor_name']});"
            )
            if output_words == state_words:
                lines.append(
                    f"    persistent_state_snapshot<{state_words}, {tensor_types[state_name]}, {output_type}>"
                    f"({state_slot['name']}, {output_buffer});"
                )
            elif output_words == 1:
                lines.append(
                    f"    persistent_state_length<{output_type}>({state_slot['cursor_name']}, {output_buffer});"
                )
            else:
                raise RuntimeError(f"HLSDAG079: PersistentStateReset output must be state-shaped or one-element length tensor")

        elif op.op_type == "Softmax":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG024: Softmax node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            axis = int((getattr(op, "attrs", {}) or {}).get("axis", -1))
            if axis < 0:
                axis += len(shape)
            if axis < 0 or axis >= len(shape):
                raise RuntimeError(f"HLSDAG025: Softmax node {op.name!r} axis {axis} is outside rank {len(shape)}")
            outer = 1
            for dim in shape[:axis]:
                outer *= dim
            axis_size = shape[axis]
            inner = 1
            for dim in shape[axis + 1:]:
                inner *= dim
            if inner == 1:
                lines.append(
                    f"    softmax_rows<{outer}, {axis_size}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[source_name]}, {output_buffer});"
                )
            else:
                lines.append(
                    f"    softmax_axis_typed<{outer}, {axis_size}, {inner}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[source_name]}, {output_buffer});"
                )
        elif op.op_type == "LayerNormalization":
            op_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            if len(op_inputs) < 1:
                raise RuntimeError(f"HLSDAG026: LayerNormalization node {op.name!r} requires an input tensor")
            source_name = op_inputs[0]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            axis = int((getattr(op, "attrs", {}) or {}).get("axis", -1))
            if axis < 0: axis += len(shape)
            if axis != len(shape) - 1:
                raise RuntimeError(f"HLSDAG027: LayerNormalization node {op.name!r} currently requires the last axis")
            cols = shape[-1]; rows = _tensor_words(graph, source_name) // cols
            epsilon = float((getattr(op, "attrs", {}) or {}).get("epsilon", 1e-5))
            if len(op_inputs) < 3:
                raise RuntimeError(f"HLSDAG028: LayerNormalization node {op.name!r} requires scale and bias tensors")
            scale_name, bias_name = op_inputs[1], op_inputs[2]
            def _norm_arg(name, role, default_value):
                if name in (getattr(graph, "constants", {}) or {}):
                    import numpy as np
                    arr = np.asarray(graph.constants[name], dtype=float).reshape(-1)
                    if arr.size != cols:
                        raise RuntimeError(f"HLSDAG029: {role} tensor {name!r} must contain {cols} values")
                    literal = ", ".join(f"{float(v):.17g}" for v in arr)
                    symbol = f"fpgai_norm_{role}_{index}"
                    lines.append(f"    static const {accumulator_type} {symbol}[{cols}] = {{ {literal} }};")
                    return symbol, accumulator_type
                if name not in tensor_to_buffer:
                    raise RuntimeError(f"HLSDAG030: {role} tensor {name!r} has no runtime buffer")
                return tensor_to_buffer[name], tensor_types.get(name, tensor_types[source_name])
            scale_arg, scale_type = _norm_arg(scale_name, "scale", 1.0)
            bias_arg, bias_type = _norm_arg(bias_name, "bias", 0.0)
            lines.append(
                f"    layer_norm_rows<{rows}, {cols}, {tensor_types[source_name]}, {scale_type}, {bias_type}, {output_type}, {accumulator_type}>("
                f"{tensor_to_buffer[source_name]}, {scale_arg}, {bias_arg}, {output_buffer}, ({accumulator_type}){epsilon:.17g});"
            )
        elif op.op_type == "RMSNorm":
            op_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            if len(op_inputs) < 2:
                raise RuntimeError(f"HLSDAG031: RMSNorm node {op.name!r} requires input and scale tensors")
            source_name, scale_name = op_inputs[0], op_inputs[1]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            axis = int((getattr(op, "attrs", {}) or {}).get("axis", -1))
            if axis < 0: axis += len(shape)
            if axis != len(shape) - 1:
                raise RuntimeError(f"HLSDAG032: RMSNorm node {op.name!r} currently requires the last axis")
            cols = shape[-1]; rows = _tensor_words(graph, source_name) // cols
            epsilon = float((getattr(op, "attrs", {}) or {}).get("epsilon", 1e-5))
            if scale_name in (getattr(graph, "constants", {}) or {}):
                import numpy as np
                arr = np.asarray(graph.constants[scale_name], dtype=float).reshape(-1)
                if arr.size != cols:
                    raise RuntimeError(f"HLSDAG033: RMSNorm scale tensor {scale_name!r} must contain {cols} values")
                literal = ", ".join(f"{float(v):.17g}" for v in arr)
                scale_arg = f"fpgai_rms_scale_{index}"
                scale_type = accumulator_type
                lines.append(f"    static const {accumulator_type} {scale_arg}[{cols}] = {{ {literal} }};")
            else:
                if scale_name not in tensor_to_buffer:
                    raise RuntimeError(f"HLSDAG034: RMSNorm scale tensor {scale_name!r} has no runtime buffer")
                scale_arg = tensor_to_buffer[scale_name]; scale_type = tensor_types.get(scale_name, tensor_types[source_name])
            lines.append(
                f"    rms_norm_rows<{rows}, {cols}, {tensor_types[source_name]}, {scale_type}, {output_type}, {accumulator_type}>("
                f"{tensor_to_buffer[source_name]}, {scale_arg}, {output_buffer}, ({accumulator_type}){epsilon:.17g});"
            )
        elif op.op_type == "RotaryEmbedding":
            op_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            if len(op_inputs) < 3:
                raise RuntimeError(f"HLSDAG037: RotaryEmbedding node {op.name!r} requires input, cosine table, and sine table")
            source_name, cos_name, sin_name = op_inputs[0], op_inputs[1], op_inputs[2]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            if len(shape) == 3 and shape[0] == 1:
                rows, cols = shape[1], shape[2]
            elif len(shape) == 2:
                rows, cols = shape
            else:
                raise RuntimeError(f"HLSDAG038: RotaryEmbedding node {op.name!r} currently requires rank-2 or batch-1 rank-3 input")
            rope_attrs = (getattr(op, "attrs", {}) or {})
            num_heads = int(rope_attrs.get("num_heads", 1) or 1)
            if num_heads <= 0 or cols % num_heads:
                raise RuntimeError(f"HLSDAG039: RotaryEmbedding node {op.name!r} requires a positive head count dividing the model width")
            head_dim = cols // num_heads
            rotary_dim = int(rope_attrs.get("rotary_dim", head_dim) or head_dim)
            if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
                raise RuntimeError(f"HLSDAG039: RotaryEmbedding node {op.name!r} requires an even rotary_dim not exceeding the per-head dimension")
            interleaved = bool(rope_attrs.get("interleaved", False))
            import numpy as np
            cos = np.asarray((getattr(graph, "constants", {}) or {}).get(cos_name), dtype=float).reshape(-1)
            sin = np.asarray((getattr(graph, "constants", {}) or {}).get(sin_name), dtype=float).reshape(-1)
            position_offset = int((getattr(op, "attrs", {}) or {}).get("position_offset", 0))
            position_input = op_inputs[3] if len(op_inputs) >= 4 else None
            if position_input is None and position_offset < 0:
                raise RuntimeError(f"HLSDAG049: RotaryEmbedding node {op.name!r} requires non-negative position_offset")
            if position_input is not None and (position_input not in tensor_to_buffer or _tensor_words(graph, position_input) != 1):
                raise RuntimeError(f"HLSDAG080: RotaryEmbedding dynamic position input must be a one-element runtime integer tensor")
            row_width = rotary_dim // 2
            if cos.size != sin.size or cos.size % row_width:
                raise RuntimeError(f"HLSDAG040: RotaryEmbedding node {op.name!r} requires matching cosine/sine tables aligned to half the rotary dimension")
            table_rows = cos.size // row_width
            if position_input is None and position_offset + rows > table_rows:
                raise RuntimeError(f"HLSDAG050: RotaryEmbedding node {op.name!r} position range exceeds cosine/sine table capacity")
            table_words = cos.size
            cos_symbol = f"fpgai_rope_cos_{index}"
            sin_symbol = f"fpgai_rope_sin_{index}"
            lines.append(f"    static const {accumulator_type} {cos_symbol}[{table_words}] = {{ {', '.join(f'{float(v):.17g}' for v in cos)} }};")
            lines.append(f"    static const {accumulator_type} {sin_symbol}[{table_words}] = {{ {', '.join(f'{float(v):.17g}' for v in sin)} }};")
            if position_input is None:
                position_expr = str(position_offset)
            else:
                pos_var = f"fpgai_rope_position_{index}"
                lines.append(f"    int {pos_var} = (int){tensor_to_buffer[position_input]}[0];")
                lines.append(f"    if ({pos_var} < 0) {pos_var} = 0;")
                lines.append(f"    if ({pos_var} > {table_rows - rows}) {pos_var} = {table_rows - rows};")
                position_expr = pos_var
            if num_heads == 1 and rotary_dim == cols:
                # Preserve the historic single-head/full-width lowering and its
                # generated-source ABI while multi-head exports use head-aware RoPE.
                lines.append(
                    f"    rotary_embedding_pairs<{rows}, {cols}, {tensor_types[source_name]}, {accumulator_type}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[source_name]}, {cos_symbol}, {sin_symbol}, {output_buffer}, {position_expr});"
                )
            else:
                lines.append(
                    f"    rotary_embedding_heads<{rows}, {cols}, {num_heads}, {rotary_dim}, {'true' if interleaved else 'false'}, "
                    f"{tensor_types[source_name]}, {accumulator_type}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[source_name]}, {cos_symbol}, {sin_symbol}, {output_buffer}, {position_expr});"
                )
        elif op.op_type == "MultiHeadAttention":
            if len(runtime_inputs) not in {3, 4}:
                raise RuntimeError(f"HLSDAG041: MultiHeadAttention node {op.name!r} requires Q, K, V and optional valid-length runtime input")
            q_name, k_name, v_name = runtime_inputs[:3]
            length_name = runtime_inputs[3] if len(runtime_inputs) == 4 else None
            q_shape = tuple(int(x) for x in graph.get_tensor(q_name).shape)
            k_shape = tuple(int(x) for x in graph.get_tensor(k_name).shape)
            v_shape = tuple(int(x) for x in graph.get_tensor(v_name).shape)
            if len(q_shape) == 3 and q_shape[0] == 1: q_shape = q_shape[1:]
            if len(k_shape) == 3 and k_shape[0] == 1: k_shape = k_shape[1:]
            if len(v_shape) == 3 and v_shape[0] == 1: v_shape = v_shape[1:]
            if len(q_shape) != 2 or len(k_shape) != 2 or len(v_shape) != 2 or k_shape != v_shape:
                raise RuntimeError(f"HLSDAG042: MultiHeadAttention node {op.name!r} requires rank-2 Q and matching rank-2 K/V tensors")
            q_seq, model = q_shape
            kv_seq, kv_model = k_shape
            attrs = getattr(op, "attrs", {}) or {}
            heads = int(attrs.get("num_heads", 1))
            kv_heads = int(attrs.get("num_kv_heads", attrs.get("num_key_value_heads", heads)))
            mode = str(attrs.get("execution_mode", "auto"))
            if mode in {"auto", "unspecified", "default", ""}:
                mode = "serialized"
            if mode not in {"serialized", "phase_shared"}:
                raise RuntimeError(f"HLSDAG043: MultiHeadAttention node {op.name!r} current HLS implementation requires serialized or phase_shared execution")
            if heads <= 0 or model % heads:
                raise RuntimeError(f"HLSDAG044: MultiHeadAttention node {op.name!r} requires model dimension divisible by num_heads")
            if kv_heads <= 0 or heads % kv_heads:
                raise RuntimeError(f"HLSDAG083: MultiHeadAttention node {op.name!r} requires num_heads divisible by num_kv_heads")
            head_dim = model // heads
            if kv_model != head_dim * kv_heads:
                raise RuntimeError(f"HLSDAG084: MultiHeadAttention node {op.name!r} K/V width must equal head_dim * num_kv_heads")
            scale = float(attrs.get("scale", 1.0 / (head_dim ** 0.5)))
            causal = bool(attrs.get("causal", True))
            masked_value = float(attrs.get("masked_value", -32.0))
            if length_name is None:
                if q_seq != kv_seq:
                    raise RuntimeError(f"HLSDAG081: MultiHeadAttention node {op.name!r} unequal Q/KV sequence lengths require a one-element valid-length input")
                if kv_heads != heads:
                    raise RuntimeError(f"HLSDAG085: MultiHeadAttention node {op.name!r} GQA currently requires cached/valid-length execution")
                lines.append(
                    f"    multi_head_attention_serialized<{q_seq}, {model}, {heads}, {tensor_types[q_name]}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[q_name]}, {tensor_to_buffer[k_name]}, {tensor_to_buffer[v_name]}, {output_buffer}, "
                    f"({accumulator_type}){scale:.17g}, {'true' if causal else 'false'}, ({accumulator_type}){masked_value:.17g});"
                )
            else:
                length_tensor = graph.get_tensor(length_name)
                if _tensor_words(graph, length_name) != 1 or not str(getattr(length_tensor, "dtype", "")).startswith(("int", "uint")):
                    raise RuntimeError(f"HLSDAG082: MultiHeadAttention node {op.name!r} valid-length input must be a one-element integer tensor")
                lines.append(
                    f"    multi_head_attention_cached_serialized<{q_seq}, {kv_seq}, {model}, {heads}, {kv_heads}, {tensor_types[q_name]}, {output_type}, {tensor_types[length_name]}, {accumulator_type}>("
                    f"{tensor_to_buffer[q_name]}, {tensor_to_buffer[k_name]}, {tensor_to_buffer[v_name]}, {tensor_to_buffer[length_name]}, {output_buffer}, "
                    f"({accumulator_type}){scale:.17g}, {'true' if causal else 'false'}, ({accumulator_type}){masked_value:.17g});"
                )
        elif op.op_type == "CausalMask":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG035: CausalMask node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            if len(shape) < 2 or shape[-1] != shape[-2]:
                raise RuntimeError(f"HLSDAG036: CausalMask node {op.name!r} requires a square score matrix on the last two dimensions")
            cols = shape[-1]; rows = _tensor_words(graph, source_name) // cols
            diagonal = int((getattr(op, "attrs", {}) or {}).get("diagonal", 0))
            masked_value = float((getattr(op, "attrs", {}) or {}).get("masked_value", -32.0))
            lines.append(
                f"    causal_mask_rows<{rows}, {cols}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                f"{tensor_to_buffer[source_name]}, {output_buffer}, {diagonal}, ({accumulator_type}){masked_value:.17g});"
            )
        elif op.op_type in {"Sub", "Div"}:
            all_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            constants = getattr(graph, "constants", {}) or {}
            if len(all_inputs) != 2:
                raise RuntimeError(f"HLSDAG051: {op.op_type} node {op.name!r} requires two inputs")
            left_name, right_name = all_inputs
            left_runtime = left_name not in constants
            right_runtime = right_name not in constants
            if left_runtime and right_runtime:
                left_words = _tensor_words(graph, left_name); right_words = _tensor_words(graph, right_name)
                if left_words == right_words == output_words:
                    fn = "sub_vec_typed" if op.op_type == "Sub" else "div_vec_typed"
                    lines.append(
                        f"    {fn}<{output_words}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}, {accumulator_type}>("
                        f"{tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer});"
                    )
                elif op.op_type == "Div":
                    left_shape = tuple(int(x) for x in graph.get_tensor(left_name).shape)
                    right_shape = tuple(int(x) for x in graph.get_tensor(right_name).shape)
                    if len(left_shape) >= 2 and right_words == left_words // left_shape[-1] and right_shape[-1:] in {(1,), ()}:
                        cols = left_shape[-1]; rows = left_words // cols
                        lines.append(
                            f"    div_rows_by_scalar_vector<{rows}, {cols}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}, {accumulator_type}>("
                            f"{tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer});"
                        )
                    else:
                        raise RuntimeError(f"HLSDAG052: Div node {op.name!r} supports equal-shaped tensors or [rows,cols]/[rows,1] broadcast")
                else:
                    raise RuntimeError(f"HLSDAG052: Sub node {op.name!r} currently requires equal-shaped runtime tensors")
            elif left_runtime and not right_runtime:
                import numpy as np
                raw_const = np.asarray(constants[right_name], dtype=float)
                if raw_const.size == 1:
                    scalar = float(raw_const.reshape(-1)[0])
                    fn = "sub_scalar_right_typed" if op.op_type == "Sub" else "div_scalar_typed"
                    lines.append(
                        f"    {fn}<{output_words}, {tensor_types[left_name]}, {output_type}, {accumulator_type}>("
                        f"{tensor_to_buffer[left_name]}, ({accumulator_type}){scalar:.17g}, {output_buffer});"
                    )
                elif op.op_type == "Sub":
                    out_shape = tuple(int(x) for x in graph.get_tensor(output_name).shape)
                    try:
                        expanded = np.broadcast_to(raw_const, out_shape).reshape(-1)
                    except ValueError as exc:
                        raise RuntimeError(f"HLSDAG053: Sub node {op.name!r} constant is not broadcast-compatible with output shape") from exc
                    if _tensor_words(graph, left_name) != output_words:
                        raise RuntimeError(f"HLSDAG053: Sub constant-broadcast path requires runtime input to match output size")
                    symbol = f"fpgai_sub_const_{index}"
                    lines.append(f"    static const {accumulator_type} {symbol}[{output_words}] = {{ {', '.join(f'{float(v):.17g}' for v in expanded)} }};")
                    lines.append(
                        f"    sub_vec_typed<{output_words}, {tensor_types[left_name]}, {accumulator_type}, {output_type}, {accumulator_type}>("
                        f"{tensor_to_buffer[left_name]}, {symbol}, {output_buffer});"
                    )
                else:
                    raise RuntimeError(f"HLSDAG053: Div constant broadcast currently requires one scalar value")
            else:
                raise RuntimeError(f"HLSDAG054: {op.op_type} current HLS profile requires the left operand to be runtime data")
        elif op.op_type == "Sqrt":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG055: Sqrt node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            if _tensor_words(graph, source_name) != output_words:
                raise RuntimeError(f"HLSDAG056: Sqrt node {op.name!r} requires equal flattened input/output sizes")
            lines.append(f"    sqrt_vec_typed<{output_words}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>({tensor_to_buffer[source_name]}, {output_buffer});")
        elif op.op_type == "Pow":
            all_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            constants = getattr(graph, "constants", {}) or {}
            if len(all_inputs) != 2 or all_inputs[0] in constants or all_inputs[1] not in constants:
                raise RuntimeError(f"HLSDAG057: Pow node {op.name!r} currently requires runtime base and constant exponent")
            import numpy as np
            exponent = np.asarray(constants[all_inputs[1]]).reshape(-1)
            if exponent.size != 1 or abs(float(exponent[0]) - 2.0) > 1e-9:
                raise RuntimeError(f"HLSDAG058: Pow node {op.name!r} current HLS profile supports exponent 2 only")
            source_name = all_inputs[0]
            lines.append(f"    square_vec_typed<{output_words}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>({tensor_to_buffer[source_name]}, {output_buffer});")
        elif op.op_type == "ReduceMean":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG059: ReduceMean node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            attrs = getattr(op, "attrs", {}) or {}
            axes = attrs.get("axes", attrs.get("dimensions", None))
            if axes is None and len(getattr(op, "inputs", []) or []) > 1:
                import numpy as np
                axes_value = (getattr(graph, "constants", {}) or {}).get(str(op.inputs[1]))
                if axes_value is not None:
                    axes = [int(x) for x in np.asarray(axes_value).reshape(-1).tolist()]
            if axes is None: axes = [-1]
            if isinstance(axes, int): axes = [axes]
            axes = [int(x) for x in axes]
            normalized = [x + len(shape) if x < 0 else x for x in axes]
            if normalized != [len(shape) - 1]:
                raise RuntimeError(f"HLSDAG060: ReduceMean node {op.name!r} current HLS profile reduces only the last axis")
            cols = shape[-1]; rows = _tensor_words(graph, source_name) // cols
            if output_words != rows:
                raise RuntimeError(f"HLSDAG061: ReduceMean node {op.name!r} output shape does not match last-axis reduction")
            lines.append(f"    reduce_mean_last_axis<{rows}, {cols}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>({tensor_to_buffer[source_name]}, {output_buffer});")
        elif op.op_type == "ReduceSum":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG086: ReduceSum node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            shape = tuple(int(x) for x in graph.get_tensor(source_name).shape)
            attrs = getattr(op, "attrs", {}) or {}
            axes = attrs.get("axes", attrs.get("dimensions", None))
            if axes is None and len(getattr(op, "inputs", []) or []) > 1:
                import numpy as np
                axes_value = (getattr(graph, "constants", {}) or {}).get(str(op.inputs[1]))
                if axes_value is not None:
                    axes = [int(x) for x in np.asarray(axes_value).reshape(-1).tolist()]
            if axes is None:
                raise RuntimeError(f"HLSDAG087: ReduceSum node {op.name!r} requires a static reduction axis")
            if isinstance(axes, int): axes = [axes]
            axes = [int(x) + len(shape) if int(x) < 0 else int(x) for x in axes]
            if len(axes) != 1 or axes[0] < 0 or axes[0] >= len(shape):
                raise RuntimeError(f"HLSDAG088: ReduceSum node {op.name!r} current HLS profile requires exactly one static axis")
            axis = axes[0]
            outer = 1
            for dim in shape[:axis]: outer *= dim
            axis_size = shape[axis]
            inner = 1
            for dim in shape[axis + 1:]: inner *= dim
            if output_words != outer * inner:
                raise RuntimeError(f"HLSDAG089: ReduceSum node {op.name!r} output shape does not match the selected reduction axis")
            lines.append(
                f"    reduce_sum_axis_typed<{outer}, {axis_size}, {inner}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                f"{tensor_to_buffer[source_name]}, {output_buffer});"
            )
        elif op.op_type == "Add":
            all_inputs = [str(x) for x in (getattr(op, "inputs", []) or [])]
            constants = getattr(graph, "constants", {}) or {}
            if len(runtime_inputs) == 1 and len(all_inputs) == 2:
                import numpy as np
                source_name = runtime_inputs[0]
                constant_name = next(name for name in all_inputs if name != source_name)
                raw_const = np.asarray(constants.get(constant_name), dtype=float)
                if raw_const.size == 1:
                    lines.append(
                        f"    add_scalar_typed<{output_words}, {tensor_types[source_name]}, {output_type}, {accumulator_type}>("
                        f"{tensor_to_buffer[source_name]}, ({accumulator_type}){float(raw_const.reshape(-1)[0]):.17g}, {output_buffer});"
                    )
                else:
                    out_shape = tuple(int(x) for x in graph.get_tensor(output_name).shape)
                    try:
                        expanded = np.broadcast_to(raw_const, out_shape).reshape(-1)
                    except ValueError as exc:
                        raise RuntimeError(f"HLSDAG009: Add node {op.name!r} constant is not broadcast-compatible with output shape") from exc
                    if _tensor_words(graph, source_name) != output_words:
                        raise RuntimeError(f"HLSDAG009: Add constant-broadcast path requires runtime input to match output size")
                    symbol = f"fpgai_add_const_{index}"
                    lines.append(f"    static const {accumulator_type} {symbol}[{output_words}] = {{ {', '.join(f'{float(v):.17g}' for v in expanded)} }};")
                    lines.append(
                        f"    add_vec_typed<{output_words}, {tensor_types[source_name]}, {accumulator_type}, {output_type}, {accumulator_type}>("
                        f"{tensor_to_buffer[source_name]}, {symbol}, {output_buffer});"
                    )
                lines.append("")
                continue
            if len(runtime_inputs) != 2:
                raise RuntimeError(f"HLSDAG009: Add node {op.name!r} requires two runtime tensors or one runtime tensor plus scalar constant")
            left_name, right_name = runtime_inputs
            left_words = _tensor_words(graph, left_name)
            right_words = _tensor_words(graph, right_name)
            if left_words != right_words or left_words != output_words:
                raise RuntimeError(f"HLSDAG010: Add node {op.name!r} requires equal flattened input/output sizes")
            qcfg = (getattr(op, "attrs", {}) or {}).get("quantized_add")
            if isinstance(qcfg, Mapping):
                lines.append(
                    f"    add_vec_quantized<{output_words}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}>("
                    f"{tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer}, "
                    f"{int(qcfg.get('left_zero', 0))}, {int(qcfg.get('left_multiplier', 1))}, {int(qcfg.get('left_shift', 0))}, "
                    f"{int(qcfg.get('right_zero', 0))}, {int(qcfg.get('right_multiplier', 1))}, {int(qcfg.get('right_shift', 0))}, "
                    f"{int(qcfg.get('output_zero', 0))}, {int(qcfg.get('qmin', -128))}, {int(qcfg.get('qmax', 127))}, "
                    f"{int(qcfg.get('rounding_mode', 0))}, {int(qcfg.get('saturation_mode', 0))});"
                )
            else:
                lines.append(
                    f"    add_vec_typed<{output_words}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}, {accumulator_type}>("
                    f"{tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer});"
                )
        else:
            raise RuntimeError(
                f"HLSDAG011: operator {op.op_type!r} is not yet supported by the branch-aware HLS emitter"
            )
        lines.append("")

    output_name = str(graph.outputs[0])
    output_buffer = tensor_to_buffer[output_name]
    output_type = tensor_types[output_name]
    output_words = _tensor_words(graph, output_name)
    output_bits = _cpp_type_bits(output_type, default=act_bits)
    if output_bits <= 0 or output_bits > axis_data_bits or axis_data_bits % output_bits != 0:
        raise ValueError(f"HLSDAG060: output tensor {output_name!r} scalar width {output_bits} is not supported by {axis_data_bits}-bit AXIS packing")
    output_per_axis = axis_data_bits // output_bits
    lines.extend([
        f"    // FPGAI_BUFFER_PROVENANCE buffer={output_buffer} tensor={output_name}",
        f"    // FPGAI_OUTPUT_SEGMENT tensor={output_name} words={output_words} bits={output_bits}",
    ])
    if output_movement.startswith("m_axi"):
        if output_bits > 32:
            raise ValueError(f"HLSDAG092: m_axi output currently supports scalar widths up to 32 bits, got {output_bits} for {output_name!r}")
        if output_movement == "m_axi_tiled":
            tile = _io_tile_size(raw_cfg, "output", output_words)
            lines.extend([
                f"    static const int FPGAI_OUTPUT_TILE_SIZE = {tile};",
                f"    {output_type} output_tile[FPGAI_OUTPUT_TILE_SIZE];",
                "#pragma HLS BIND_STORAGE variable=output_tile type=ram_1p impl=bram",
                f"    // m_axi tiled output export: {output_buffer} -> output_tile -> output_mem.",
                f"    for (int tile_base = 0; tile_base < {output_words}; tile_base += FPGAI_OUTPUT_TILE_SIZE) {{",
                f"        int tile_count = ((tile_base + FPGAI_OUTPUT_TILE_SIZE) <= {output_words}) ? FPGAI_OUTPUT_TILE_SIZE : ({output_words} - tile_base);",
                "        for (int lane = 0; lane < FPGAI_OUTPUT_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                f"            if (lane < tile_count) output_tile[lane] = {output_buffer}[tile_base + lane];",
                "        }",
                "        for (int lane = 0; lane < FPGAI_OUTPUT_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                "            if (lane < tile_count) {",
                f"                ap_uint<{output_bits}> raw = output_tile[lane].range({output_bits - 1}, 0);",
                "                output_mem[tile_base + lane] = raw;",
                "            }",
                "        }",
                "    }",
            ])
        else:
            lines.extend([
                f"    // m_axi full output export: {output_buffer} -> output_mem.",
                f"    for (int index = 0; index < {output_words}; ++index) {{",
                "#pragma HLS PIPELINE II=1",
                f"        ap_uint<{output_bits}> raw = {output_buffer}[index].range({output_bits - 1}, 0);",
                "        output_mem[index] = raw;",
                "    }",
            ])
    elif output_movement == "axi_stream_tiled":
        tile = _io_tile_size(raw_cfg, "output", output_words)
        lines.extend([
            f"    static const int FPGAI_AXIS_OUTPUT_TILE_SIZE = {tile};",
            f"    {output_type} output_tile[FPGAI_AXIS_OUTPUT_TILE_SIZE];",
            "#pragma HLS BIND_STORAGE variable=output_tile type=ram_1p impl=bram",
            f"    for (int tile_base = 0; tile_base < {output_words}; tile_base += FPGAI_AXIS_OUTPUT_TILE_SIZE) {{",
            f"        int tile_count = ((tile_base + FPGAI_AXIS_OUTPUT_TILE_SIZE) <= {output_words}) ? FPGAI_AXIS_OUTPUT_TILE_SIZE : ({output_words} - tile_base);",
            "        for (int lane = 0; lane < FPGAI_AXIS_OUTPUT_TILE_SIZE; ++lane) {",
            "#pragma HLS PIPELINE II=1",
            f"            if (lane < tile_count) output_tile[lane] = {output_buffer}[tile_base + lane];",
            "        }",
            "        for (int lane = 0; lane < FPGAI_AXIS_OUTPUT_TILE_SIZE; ++lane) {",
            "#pragma HLS PIPELINE II=1",
            "            if (lane < tile_count) {",
            "                axis_t packet; packet.data = 0; packet.keep = -1; packet.strb = -1; packet.last = 0;",
            f"                fpgai_pack_axis_value<{output_type}, {output_bits}>(packet, output_tile[lane], 0);",
            f"                packet.last = ((tile_base + lane + 1) >= {output_words}) ? 1 : 0;",
            "                out_stream.write(packet);",
            "            }",
            "        }",
            "    }",
        ])
    else:
        lines.extend([
            f"    for (int base = 0; base < {output_words}; base += {output_per_axis}) {{",
            "#pragma HLS PIPELINE II=1",
            "        axis_t packet;",
            "        packet.data = 0; packet.keep = -1; packet.strb = -1; packet.last = 0;",
            f"        for (int lane = 0; lane < {output_per_axis}; ++lane) {{",
            "#pragma HLS UNROLL",
            "            int index = base + lane;",
            f"            if (index < {output_words}) fpgai_pack_axis_value<{output_type}, {'FPGAI_ACT_BITS' if output_bits == act_bits else output_bits}>(packet, {output_buffer}[index], lane);",
            "        }",
            f"        packet.last = (base + {output_per_axis} >= {output_words}) ? 1 : 0;",
            "        out_stream.write(packet);",
            "    }",
        ])
    lines.extend(["}", ""])
    source = "\n".join(lines)
    if normalized_weights_mode == "embedded":
        # Keep DAG embedded-weight storage on the same physical contract as
        # the linear emitter.  Global const arrays in fpgai_params.cpp are
        # only the compile-time initialization image; Vitis cannot apply
        # file-scope BIND_STORAGE pragmas to them.  Materialize function-scope
        # static arrays and bind them explicitly to the user-selected BRAM or
        # URAM implementation.
        raw = dict(raw_cfg or {})
        memory = raw.get("memory") if isinstance(raw.get("memory"), Mapping) else {}
        storage_cfg = memory.get("storage") if isinstance(memory.get("storage"), Mapping) else {}
        storage = str(
            storage_cfg.get("weights")
            or memory.get("weight_storage")
            or "bram"
        ).strip().lower().replace("-", "_")
        storage_aliases = {
            "embedded": "bram",
            "on_chip": "bram",
            "onchip": "bram",
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "uram": "uram",
        }
        impl = storage_aliases.get(storage)
        if impl is not None:
            from fpgai.backends.hls.emit.top_cpp import _fpgai_insert_static_weight_block
            source = _fpgai_insert_static_weight_block(source, graph, impl=impl)

    if normalized_weights_mode == "ddr_tiled":
        # Reuse the existing DDR-tiled weight ABI and Dense/Conv tile helpers.
        # The DAG Conv path stores feature maps internally in HWC-flat order,
        # so request the HWC layout variant of the shared Conv helper.
        from fpgai.backends.hls.emit.top_cpp import (
            _fpgai_insert_ddr_tiled_pragmas,
            _fpgai_rewrite_conv_calls_for_ddr_tiled,
            _fpgai_rewrite_dense_calls_for_ddr_tiled,
            _fpgai_rewrite_runtime_signature,
        )

        source = _fpgai_rewrite_runtime_signature(source, top_name=top_name, mode="ddr_tiled")
        source = _fpgai_insert_ddr_tiled_pragmas(source)
        source = _fpgai_rewrite_conv_calls_for_ddr_tiled(source, graph, layout="hwc")
        source = _fpgai_rewrite_dense_calls_for_ddr_tiled(source, graph)
        source = (
            "// Requested DAG weights mode: ddr_tiled\n"
            "// DDR-tiled DAG inference keeps full Conv/Dense weights in weights_mem and materializes only tile-sized on-chip buffers.\n"
            + source
        )
    elif normalized_weights_mode != "embedded":
        # Reuse the same runtime-weight ABI and import/export implementation as
        # the linear inference emitter. The generated DAG compute body still
        # consumes W*/B* symbols, while command mode 1 materializes the payload
        # into BRAM/URAM and optional mode 2 exports it again.
        from fpgai.backends.hls.emit.top_cpp import (
            _fpgai_insert_runtime_helpers,
            _fpgai_insert_runtime_load_block,
            _fpgai_rewrite_runtime_signature,
        )

        source = _fpgai_insert_runtime_helpers(source)
        source = _fpgai_rewrite_runtime_signature(
            source, top_name=top_name, mode=normalized_weights_mode
        )
        raw = dict(raw_cfg or {})
        memory = raw.get("memory") if isinstance(raw.get("memory"), Mapping) else {}
        storage = str(
            ((memory.get("storage") or {}).get("weights") if isinstance(memory.get("storage"), Mapping) else None)
            or memory.get("weight_storage")
            or "bram"
        ).strip().lower()
        requested = str(((raw.get("weights") or {}).get("mode") if isinstance(raw.get("weights"), Mapping) else "") or "").strip().lower()
        resolved_semantics = ""
        if requested == "import_export" and storage in {"bram", "uram"}:
            resolved_semantics = f"{storage}_import_export_full"
        elif requested == "import" and storage in {"bram", "uram"}:
            resolved_semantics = f"{storage}_import_full"
        source = _fpgai_insert_runtime_load_block(
            source,
            graph,
            mode=normalized_weights_mode,
            resolved_semantics=resolved_semantics,
        )
        source = (
            f"// Requested DAG weights mode: {normalized_weights_mode}\n"
            "// Runtime parameter movement uses the shared FPGAI inference weight ABI.\n"
            + source
        )

    if input_movement.startswith("m_axi") or output_movement.startswith("m_axi"):
        from fpgai.backends.hls.emit.top_cpp import _fpgai_rewrite_signature_for_m_axi_io
        source = _fpgai_rewrite_signature_for_m_axi_io(
            source,
            input_m_axi=input_movement.startswith("m_axi"),
            output_m_axi=output_movement.startswith("m_axi"),
        )
    source = _rewrite_external_state_signature(source, top_name=top_name, slots=external_persistent_slots)

    # Keep DAG-generated inference tops on the same generated-source contract
    # as the linear emitter.  These markers are not cosmetic-only: reporting
    # uses them to verify that the requested movement/readability policy was
    # materially reflected in the final HLS artifact.
    if (
        input_movement == "axi_stream_tiled"
        or output_movement == "axi_stream_tiled"
    ) and "FPGAI AXI-stream tiled input/output movement" not in source:
        source = "// FPGAI AXI-stream tiled input/output movement.\n" + source

    from fpgai.backends.hls.emit.top_cpp import _fpgai_readability_banner

    banner = _fpgai_readability_banner(
        {
            "raw_cfg": dict(raw_cfg or {}),
            "weights_mode": normalized_weights_mode,
        },
        kind="inference",
    )
    if banner and "FPGAI generated HLS top" not in source[:512]:
        source = banner + source
    return source
