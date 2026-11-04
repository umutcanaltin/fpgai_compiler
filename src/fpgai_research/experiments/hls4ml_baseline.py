"""Prepare a hls4ml comparator from the identical ONNX model without vendor execution."""
from __future__ import annotations
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
import re


def fixed_precision(bits: int, integer: int) -> str:
    if type(bits) is not int or type(integer) is not int or not (2 <= bits <= 32 and 1 <= integer <= bits):
        raise ValueError("Fixed-point width/integer bits must be integers in range 2..32")
    return f"ap_fixed<{bits},{integer},AP_RND,AP_SAT>"


def _attest_private_inventory(*, repository: str | Path, inventory_dir: str | Path,
                              model_kind: str, source: Path, model_sha256: str) -> str:
    """Reaudit the live original pair before binding a candidate source to it.

    An inventory JSON file by itself is not trusted: the saved complete
    result must match a fresh inspection of BOTH original trained ONNX
    files and the paper-suite copies using the current ONNX library.
    """
    from fpgai_research.experiments import hls4ml_onnx_inventory as inventory

    if model_kind not in inventory.KINDS:
        raise ValueError("Inventory-bound model kind must be mlp or cnn")
    root = Path(repository)
    expected = root / inventory._DEFAULT_TRAINED[model_kind] / "model.onnx"
    if expected.is_symlink() or not expected.is_file() or source != expected.resolve(strict=True):
        raise ValueError("Inventory-bound conversion must use original trained model.onnx")
    snapshot = Path(inventory_dir) / "inventory.json"
    if snapshot.is_symlink() or not snapshot.is_file() or snapshot.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Missing or unsafe private ONNX inventory")
    raw = snapshot.read_bytes()
    try:
        saved = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Malformed private ONNX inventory") from exc
    live = inventory.collect_onnx_inventory(root)
    if (saved != live or live.get("schema") != inventory.SCHEMA
            or live.get("model_count") != 2
            or live.get("external_qor_comparison_eligible") is not False
            or {r.get("model") for r in live["models"]} != set(inventory.KINDS)):
        raise ValueError("Private ONNX inventory stale, altered, or incomplete")
    row = next(r for r in live["models"] if r["model"] == model_kind)
    if (row.get("onnx_sha256") != model_sha256
            or row.get("trained_onnx_matches_assembled") is not True
            or row.get("graph_checked_by_onnx") is not True):
        raise ValueError("ONNX inventory does not attest the requested candidate source")
    return sha256(raw).hexdigest()


def _attest_derived_e0(*, repository: str | Path, candidate_dir: str | Path,
                       full_e0_dir: str | Path, kind: str, model_sha256: str) -> dict:
    """Recheck exact Gemm-free ONNX and qualified 10000-image E0 lineage.

    This is an original→derived SOFTWARE proof obligation only, not
    trained original ONNX→native Vitis/Vivado source/weight lineage.
    """
    from fpgai_research.experiments.hls4ml_candidate_heldout import (
        attest_candidate, _private_json, SCHEMA as E0_SCHEMA, COUNT,
    )
    from fpgai_research.experiments.mnist_heldout_paper import collect_heldout_paper
    from fpgai_research.experiments.hls4ml_onnx_inventory import KINDS
    if kind not in KINDS:
        raise ValueError("Expected original MLP or CNN kind")
    record = attest_candidate(repository, candidate_dir, kind)
    if record["candidate_sha256"] != model_sha256:
        raise ValueError("Conversion input does not match source-attested derived ONNX")
    report = _private_json(Path(full_e0_dir) / "heldout.json")
    metrics = report.get("numeric_comparison")
    original = collect_heldout_paper(repository)
    expected = next((r for r in original["models"] if r["model"] == kind), None)
    if (report.get("schema") != E0_SCHEMA
            or report.get("model_kind") != kind
            or report.get("evidence") != "E0-full-official-MNIST-original-vs-derived-ONNX"
            or report.get("source_onnx_sha256") != record["source_sha256"]
            or report.get("candidate_onnx_sha256") != model_sha256
            or report.get("hls4ml_e1_e2_e3_e4_qualified") is not False
            or report.get("native_vendor_tools_executed") is not False
            or report.get("external_qor_comparison_eligible") is not False
            or report.get("original_trained_to_native_hls_weight_lineage_verified") is not False
            or not isinstance(metrics, dict)
            or metrics.get("official_test_images") != COUNT
            or metrics.get("candidate_onnx_software_e0_qualified") is not True
            or metrics.get("top1_prediction_disagreements") != 0
            or metrics.get("native_fpga_full_test_accuracy_measured") is not False
            or expected is None
            or metrics.get("original_correct_top1") != expected["correct_top1_onnxruntime"]
            or metrics.get("candidate_correct_top1") != expected["correct_top1_onnxruntime"]):
        raise ValueError("Missing or mismatched full software E0 candidate qualification")
    return {
        "original_source_onnx_sha256": record["source_sha256"],
        "derived_candidate_onnx_sha256": record["candidate_sha256"],
        "full_e0_receipt_sha256": sha256(
            (Path(full_e0_dir) / "heldout.json").read_bytes()).hexdigest(),
        "full_e0_selected_candidate_qualified": True,
        "original_to_native_e3_e4_weight_identity_verified": False,
        "model_kind": kind,
    }


def annotate_initializer_shapes_for_hls4ml(onnx_model):
    """Make a name- and shape-compatible in-memory ONNX view for hls4ml 1.3.0.

    hls4ml's get_input_shape() reads graph.value_info/input/output and
    *does not* consult graph.initializer; onnx.shape_inference alone does not
    need to populate value_info for constant initializers. Explicit static
    float32 value info for ALL initializers bridges that parser assumption.

    The v1.3.0 frontend also replaces periods in Constant *layer*
    names without updating input edges. This view consistently aliases
    such initializer names and all references. It fails on collisions.

    The immutable original and Gemm-normalized ONNX files are NOT rewritten.
    ONNX node-name metadata containing C++-unsafe characters is also
    replaced by deterministic, collision-checked valid identifiers.
    Unsafe ONNX tensor identities are rewritten in the same temporary
    view across graph and producer/consumer references; trained source
    weights and full E0 original/candidate files are never rewritten.
    Constant bytes, computational operators, graph edges and numeric
    semantics are preserved. This conversion view is not E0, E2 or hardware proof.
    """
    import onnx
    from onnx import TensorProto, helper

    result = onnx.ModelProto()
    result.CopyFrom(onnx_model)
    graph = result.graph
    if graph.sparse_initializer:
        raise ValueError("Sparse ONNX initializer incompatible with bounded hls4ml adapter")
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in graph.node for a in node.attribute):
        raise ValueError("Nested ONNX graphs require an independent hls4ml normalization")
    # hls4ml v1.3.0 onnx_to_hls.py calls replace_char_inconsitency()
    # on initializers (dots -> underscores) BUT leaves node.input edges
    # untouched. That breaks Constant -> Merge edges when a trained
    # bias is named, for example, dense.0.bias.
    #
    # Rename only affected constant tensor identities AND every graph
    # reference to them, on an in-memory copy. Refuse even potential
    # name collisions; never change a model file or tensor bytes.
    all_names = {name for node in graph.node for name in (*node.input, *node.output)
                 if name}
    all_names.update(t.name for t in graph.initializer)
    all_names.update(v.name for v in (*graph.input, *graph.output, *graph.value_info))
    aliases = {}
    for tensor in graph.initializer:
        if not tensor.name:
            raise ValueError("Unnamed ONNX initializer is unsafe for hls4ml")
        normalized = tensor.name.replace(".", "_")
        if normalized != tensor.name:
            if normalized in all_names or normalized in aliases.values():
                raise ValueError("Initializer alias collides with another ONNX tensor")
            aliases[tensor.name] = normalized
    if aliases:
        for tensor in graph.initializer:
            if tensor.name in aliases:
                tensor.name = aliases[tensor.name]
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name in aliases:
                    node.input[i] = aliases[name]
        for value in (*graph.input, *graph.output, *graph.value_info):
            if value.name in aliases:
                value.name = aliases[value.name]
    # hls4ml 1.3.0 uses ONNX node.name in generated C++ typedefs,
    # including activation lookup-table types. The original PyTorch
    # exporter emits names like /network/network.2/Relu and /Softmax,
    # which parse as ONNX yet cause the actual Vitis CSim GCC compiler
    # to reject invalid names such as /Softmax_table_t.
    #
    # A node's name is descriptive metadata, not a tensor identifier.
    # Rewrite it ONLY in the disposable hls4ml view; do not touch
    # node inputs/outputs, operator types, attributes or weight values.
    # Preserve already-safe MatMul names so independently configured
    # ReuseFactor controls continue to address the same layers.
    cpp_keywords = {
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand",
        "bitor", "bool", "break", "case", "catch", "char", "char16_t",
        "char32_t", "class", "compl", "const", "constexpr", "const_cast",
        "continue", "decltype", "default", "delete", "do", "double",
        "dynamic_cast", "else", "enum", "explicit", "export", "extern",
        "false", "float", "for", "friend", "goto", "if", "inline", "int",
        "long", "mutable", "namespace", "new", "noexcept", "not",
        "not_eq", "nullptr", "operator", "or", "or_eq", "private",
        "protected", "public", "register", "reinterpret_cast", "return",
        "short", "signed", "sizeof", "static", "static_assert",
        "static_cast", "struct", "switch", "template", "this",
        "thread_local", "throw", "true", "try", "typedef", "typeid",
        "typename", "union", "unsigned", "using", "virtual", "void",
        "volatile", "wchar_t", "while", "xor", "xor_eq",
    }
    all_tensor_names = {
        name for node in graph.node for name in (*node.input, *node.output)
        if name
    }
    all_tensor_names.update(t.name for t in graph.initializer)
    all_tensor_names.update(
        v.name for v in (*graph.input, *graph.output, *graph.value_info)
    )
    original_node_names = [node.name for node in graph.node if node.name]
    if len(set(original_node_names)) != len(original_node_names):
        raise ValueError("Duplicated ONNX node names require independent review")
    chosen = set()
    node_names_changed = 0
    previously_unnamed = 0
    for index, node in enumerate(graph.node):
        previous = node.name
        valid = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", previous))
        if valid and previous not in cpp_keywords:
            proposed = previous
        else:
            # Stable short generated name also avoids leaking private
            # PyTorch hierarchy names into C++ identifiers.
            proposed = f"hls4ml_node_{index}_{node.op_type}"
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", proposed)
                or proposed in cpp_keywords
                or (proposed != previous and proposed in all_tensor_names)
                or proposed in chosen
                or (proposed != previous and proposed in original_node_names)):
            raise ValueError("hls4ml ONNX node identifier collides with existing graph identity")
        chosen.add(proposed)
        if proposed != previous:
            node.name = proposed
            node_names_changed += 1
            previously_unnamed += int(not previous)
    # A separate hls4ml v1.3.0 limitation is that it uses ONNX
    # *tensor* identities in C++ precision and variable names. Sanitizing
    # node.name alone does not guarantee compilable defines.h. Convert all
    # malformed tensor identities consistently, including inputs/outputs,
    # weight initializers and intermediate producer-consumer edges.
    if graph.quantization_annotation:
        raise ValueError("Quantized ONNX tensor annotations require independent review")
    tensor_identities = set()
    tensor_order = []
    def remember_tensor(name):
        if name and name not in tensor_identities:
            tensor_order.append(name)
            tensor_identities.add(name)
    for info in (*graph.input, *graph.output, *graph.value_info):
        remember_tensor(info.name)
    for tensor in graph.initializer:
        remember_tensor(tensor.name)
    for node in graph.node:
        for name in (*node.input, *node.output):
            remember_tensor(name)
    tensor_aliases = {}
    for index, name in enumerate(tensor_order):
        if (re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                and name not in cpp_keywords):
            continue
        candidate_name = f"hls4ml_tensor_{index}"
        if (candidate_name in tensor_identities or candidate_name in chosen
                or candidate_name in tensor_aliases.values()):
            raise ValueError("C++ tensor name alias collides with existing graph identity")
        tensor_aliases[name] = candidate_name
    for info in (*graph.input, *graph.output, *graph.value_info):
        if info.name in tensor_aliases:
            info.name = tensor_aliases[info.name]
    for tensor in graph.initializer:
        if tensor.name in tensor_aliases:
            tensor.name = tensor_aliases[tensor.name]
    for node in graph.node:
        for sequence in (node.input, node.output):
            for index, name in enumerate(sequence):
                if name in tensor_aliases:
                    sequence[index] = tensor_aliases[name]

    known = {}
    for info in (*graph.input, *graph.output, *graph.value_info):
        if not info.name or info.name in known:
            raise ValueError("Duplicate/empty typed tensor identity in conversion view")
        known[info.name] = info
    changes = []
    for tensor in graph.initializer:
        if (not tensor.name or tensor.data_type != TensorProto.FLOAT
                or not tensor.dims or any(dim < 1 for dim in tensor.dims)
                or tensor.data_location == TensorProto.EXTERNAL
                or tensor.external_data):
            raise ValueError("Only embedded static nonempty float32 hls4ml constants are supported")
        expected = tuple(int(dim) for dim in tensor.dims)
        existing = known.get(tensor.name)
        if existing is not None:
            ttype = existing.type.tensor_type
            actual = tuple(int(dim.dim_value) if dim.HasField("dim_value") else None
                           for dim in ttype.shape.dim)
            if ttype.elem_type != TensorProto.FLOAT or actual != expected:
                raise ValueError("Existing ONNX constant value_info disagrees with initializer")
            continue
        graph.value_info.append(
            helper.make_tensor_value_info(tensor.name, TensorProto.FLOAT, list(expected)))
        changes.append({"shape":list(expected),"rank":len(expected)})
    onnx.checker.check_model(result)
    # Presence and dimensionality are auditable without revealing trained
    # tensor identities or raw floating-point values in the receipt.
    return result, {"schema":"fpgai-research/hls4ml-constant-shape-view/v2",
                    "metadata_only":False,
                    "tensor_identity_aliases_only":True,
                    "numeric_semantics_modified":False,
                    "original_onnx_source_unchanged":True,
                    "renamed_initializer_identity_count":len(aliases),
                    "renamed_onnx_tensor_identity_count":len(tensor_aliases),
                    "renamed_onnx_node_name_count":node_names_changed,
                    "previously_unnamed_onnx_node_count":previously_unnamed,
                    "added_initializer_value_info_count":len(changes),
                    "added_initializer_shapes":changes}



def reject_known_invalid_hls4ml_cpp_typedefs(files):
    """Block an observed hls4ml v1.3 ONNX-name C++ codegen failure before E1 export.

    This is a narrow syntactic check, NOT a substitute for CSim compilation.
    No tensor, layer or private identifier names appear in diagnostics.
    """
    from fpgai_research.experiments.hls4ml_cpp_compat import (
        UNSAFE_TYPE, is_model_type_header,
    )
    invalid = UNSAFE_TYPE
    for path in files:
        # Do not read or reject bundled Vitis support headers containing
        # valid C++ division (e.g. ap_types/ap_int_base.h:123).
        if (path.is_file() and is_model_type_header(path)
                and path.stat().st_size <= 16 * 1024 * 1024):
            text = path.read_text(encoding="utf-8")
            match = invalid.search(text)
            if match:
                # Keep source contents and private model names out of the
                # error. A line number still lets us locate whether the
                # remaining cause is a node name, tensor name or an
                # overbroad static pattern.
                line_number = text.count("\n", 0, match.start()) + 1
                raise ValueError(
                    "Generated hls4ml C++ typedef has an unsafe source identifier "
                    f"(generated header line {line_number}); refusing E1. "
                    "Original ONNX and previous receipts are unchanged.")
    return True


def apply_single_layer_control(config, request):
    """Apply one explicitly typed hls4ml layer setting to a fresh config.

    This changes source-level intent only. It must be paired with a
    separate matched baseline and audited after ModelGraph conversion.
    Unknown/unsupported backend attributes fail closed.
    """
    if (not isinstance(request, dict) or len(request) != 1
            or not isinstance(config, dict)
            or not isinstance(config.get("LayerName"), dict)):
        raise ValueError("Require one named hls4ml layer with installed name-level controls")
    name, settings = next(iter(request.items()))
    if (not isinstance(name, str) or not name
            or not isinstance(settings, dict) or len(settings) != 1
            or name not in config["LayerName"]
            or not isinstance(config["LayerName"][name], dict)):
        raise ValueError("Require one real hls4ml layer and one control")
    knob, value = next(iter(settings.items()))
    accepted = {"ReuseFactor", "Precision", "Strategy", "ParallelizationFactor"}
    if knob not in accepted or knob not in config["LayerName"][name]:
        raise ValueError("Layer does not advertise this hls4ml configurable attribute")
    if knob in ("ReuseFactor", "ParallelizationFactor"):
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError("Positive integer parallelism/reuse control required")
    elif knob == "Strategy":
        if value not in ("Latency", "Resource"):
            raise ValueError("Unsupported hls4ml optimization strategy")
    elif knob == "Precision":
        if not isinstance(value, str) or not re.fullmatch(
                r"ap_fixed<[0-9]{1,2},[0-9]{1,2},AP_RND,AP_SAT>", value):
            raise ValueError("Only explicit bounded signed fixed-point precision accepted")
        numbers = [int(x) for x in re.findall(r"\d+", value)]
        if not 2 <= numbers[0] <= 32 or not 1 <= numbers[1] <= numbers[0]:
            raise ValueError("Invalid precision width/integer bits")
    config["LayerName"][name][knob] = value
    return {"layer": name, "attribute": knob, "requested_value": value,
            "native_effective_attribute_and_hardware_not_proven": True}


def prepare_baseline(*, model_path: str | Path, output_dir: str | Path,
                     bits: int = 16, integer: int = 6, reuse_factor: int = 1,
                     part: str = "xck26-sfvc784-2LV-c", clock_ns: float = 5.0,
                     inventory_dir: str | Path | None = None,
                     repository: str | Path | None = None,
                     model_kind: str | None = None,
                     candidate_dir: str | Path | None = None,
                     full_e0_dir: str | Path | None = None,
                     config_granularity: str = "model",
                     layer_reuse: dict[str, int] | None = None,
                     failure_diagnostic_dir: str | Path | None = None,
                     name_config_policy: str = "generated",
                     layer_controls: dict[str, dict[str, object]] | None = None) -> dict:
    """Generate a private E1 candidate only, with no CSim, HLS synth or Vivado.

    No receipt is published until at least one actual generated C++ source
    exists and its file-set digest is recorded. Failed writes leave no public
    output directory; the original ONNX is never intentionally modified.
    """
    if config_granularity not in ("model", "name"):
        raise ValueError("Only model-wide and per-layer-name hls4ml controls are supported")
    if name_config_policy not in ("generated", "reuse-only",
                                   "selected-reuse-only"):
        raise ValueError("Unsupported name-level configuration policy")
    if name_config_policy != "generated" and config_granularity != "name":
        raise ValueError("Reuse-only policy requires name-level configuration")
    if layer_reuse is not None:
        if (config_granularity != "name" or not isinstance(layer_reuse, dict)
                or not layer_reuse or len(layer_reuse) > 128
                or any(not isinstance(key, str) or not key or len(key) > 256
                       or type(value) is not int or not 1 <= value <= 4096
                       for key, value in layer_reuse.items())):
            raise ValueError("Invalid explicit per-layer reuse controls")
    if layer_controls is not None and (
            config_granularity != "name" or name_config_policy != "generated"
            or layer_reuse is not None):
        raise ValueError("Single-layer intervention requires generated name-level config and no reuse-only override")
    precision = fixed_precision(bits, integer)
    if type(reuse_factor) is not int or not (1 <= reuse_factor <= 4096):
        raise ValueError("Reuse factor must be an integer in 1..4096")
    if not isinstance(part, str) or not re.fullmatch(r"[A-Za-z0-9_.+\-]+", part):
        raise ValueError("Invalid FPGA part")
    if type(clock_ns) not in (int, float) or not isfinite(clock_ns) or clock_ns <= 0:
        raise ValueError("Clock period must be positive and finite")
    model_input = Path(model_path)
    if model_input.is_symlink() or not model_input.is_file():
        raise ValueError("Baseline ONNX source must be a regular non-linked file")
    source = model_input.resolve(strict=True)
    if source.suffix.lower() != ".onnx":
        raise ValueError("Expected one ONNX model")
    model_sha256 = sha256(source.read_bytes()).hexdigest()
    if candidate_dir is not None:
        if (inventory_dir is not None or repository is None or model_kind is None
                or full_e0_dir is None):
            raise ValueError("Derived baseline requires repository, kind, full E0 and no original inventory option")
        expected_candidate = Path(candidate_dir) / "candidate.onnx"
        if (expected_candidate.is_symlink() or not expected_candidate.is_file()
                or expected_candidate.resolve(strict=True) != source):
            raise ValueError("Derived baseline input must equal original immutable candidate.onnx")
        e0_binding = _attest_derived_e0(
            repository=repository, candidate_dir=candidate_dir,
            full_e0_dir=full_e0_dir, kind=model_kind, model_sha256=model_sha256)
        inventory_sha256 = None
    else:
        if full_e0_dir is not None:
            raise ValueError("Full E0 evidence requires derived candidate")
        e0_binding = None
        if inventory_dir is None:
            if repository is not None or model_kind is not None:
                raise ValueError("Inventory repository/model-kind requires inventory_dir")
            inventory_sha256 = None
        else:
            if repository is None or model_kind is None:
                raise ValueError("Inventory-bound conversion requires repository and model-kind")
            inventory_sha256 = _attest_private_inventory(
                repository=repository, inventory_dir=inventory_dir,
                model_kind=model_kind, source=source, model_sha256=model_sha256)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to mix hls4ml runs: output directory already exists")

    try:
        import onnx
        import hls4ml
    except ImportError as exc:
        raise RuntimeError("Install matching pinned ONNX and hls4ml releases") from exc
    if e0_binding is not None:
        from fpgai_research.experiments.hls4ml_frontend_readiness import (
            collect_frontend_readiness, PINNED_HLS4ML_VERSION,
        )
        frontend = collect_frontend_readiness(repository, candidate_dir, model_kind)
        if (str(hls4ml.__version__) != PINNED_HLS4ML_VERSION
                or frontend.get("source_only_conversion_attempt_ready") is not True):
            raise ValueError(
                "Derived hls4ml E1 attempt blocked by installed ONNX frontend "
                + ", ".join(frontend.get("blockers", [])))
    source_model = onnx.load(str(source))
    onnx.checker.check_model(source_model)
    hls4ml_view = None
    if e0_binding is not None:
        source_model, hls4ml_view = annotate_initializer_shapes_for_hls4ml(source_model)
    config = hls4ml.utils.config_from_onnx_model(
        source_model, granularity=config_granularity, default_precision=precision,
        default_reuse_factor=reuse_factor, backend="Vitis")
    if not isinstance(config, dict) or not isinstance(config.get("Model"), dict):
        raise ValueError("Unexpected hls4ml ONNX config structure")
    name_config_diagnostic = None
    if config_granularity == "model":
        if any(key in config for key in ("LayerName", "LayerType")):
            raise ValueError("Unexpected per-layer controls in global-only baseline")
    else:
        names = config.get("LayerName")
        if not isinstance(names, dict) or not names or "LayerType" in config:
            raise ValueError("hls4ml did not generate per-layer-name configuration")
        if any(not isinstance(name, str) or not name
               or not isinstance(values, dict)
               for name, values in names.items()):
            raise ValueError("Malformed hls4ml per-layer config inventory")
        # hls4ml 1.3.0 config_from_onnx_model(granularity="name")
        # expands *every configurable precision/type/strategy* attribute.
        # Consequently RF1 at name granularity is not numerically equivalent
        # to Model-level RF1. An explicit reuse-only control removes the
        # generated per-layer non-reuse overrides, letting the Model-level
        # Precision and other defaults continue to apply.
        name_config_diagnostic = {
            "policy": name_config_policy,
            "generated_layer_count": len(names),
            "generated_layer_precision_override_count":
                sum(int("Precision" in item) for item in names.values()),
            "generated_non_reuse_key_count":
                sum(sum(key != "ReuseFactor" for key in item) for item in names.values()),
            "non_reuse_key_count_removed": 0,
            "reuse_configurable_layer_count":
                sum(int("ReuseFactor" in item) for item in names.values()),
        }
        if name_config_policy == "reuse-only":
            name_config_diagnostic["non_reuse_key_count_removed"] = (
                name_config_diagnostic["generated_non_reuse_key_count"])
            config["LayerName"] = {
                name: {"ReuseFactor": item["ReuseFactor"]}
                for name, item in names.items() if "ReuseFactor" in item
            }
            names = config["LayerName"]
        if name_config_policy == "selected-reuse-only":
            # Validate names/attribute support against the *original*
            # pinned-backend config BEFORE discarding its implicit settings.
            if not layer_reuse or len(layer_reuse) != 1:
                raise ValueError("Selected-reuse-only requires exactly one layer request")
            for selected, factor in layer_reuse.items():
                if selected not in names or "ReuseFactor" not in names[selected]:
                    raise ValueError("Selected layer lacks configurable ReuseFactor")
                if type(factor) is not int or factor < 1:
                    raise ValueError("Selected layer ReuseFactor must be a positive integer")
            name_config_diagnostic["non_reuse_key_count_removed"] = (
                name_config_diagnostic["generated_non_reuse_key_count"])
            name_config_diagnostic["implicit_reuse_entries_removed"] = (
                name_config_diagnostic["reuse_configurable_layer_count"] - 1)
            config["LayerName"] = {
                selected: {"ReuseFactor": factor}
                for selected, factor in layer_reuse.items()
            }
            names = config["LayerName"]
        for name, value in (layer_reuse or {}).items():
            if name not in names:
                raise ValueError("Unknown hls4ml layer name: " + name)
            if "ReuseFactor" not in names[name]:
                raise ValueError("Requested layer does not expose configurable ReuseFactor: " + name)
            names[name]["ReuseFactor"] = value
    single_layer_intervention = (
        apply_single_layer_control(config, layer_controls)
        if layer_controls is not None else None
    )
    config["Model"]["Precision"] = {"default": precision}
    config["Model"]["ReuseFactor"] = reuse_factor

    # hls4ml may normalize its configuration while creating a ModelGraph.
    # Preserve the original explicit user-facing contract separately from
    # the post-conversion config in the immutable E1 receipt.
    from copy import deepcopy
    requested_hls_config = deepcopy(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-", dir=output.parent) as temp:
        # Let hls4ml create the project directory; a failed conversion can only
        # affect a private temporary location, never a final comparison receipt.
        stage = Path(temp) / "project"
        model = hls4ml.converters.convert_from_onnx_model(
            source_model, hls_config=config, output_dir=str(stage),
            project_name="fpgairesearch_baseline", backend="Vitis",
            io_type="io_parallel", part=part, clock_period=clock_ns)
        model.write()
        if sha256(source.read_bytes()).hexdigest() != model_sha256:
            raise ValueError("Baseline ONNX changed during hls4ml conversion; refusing evidence receipt")
        if e0_binding is not None and _attest_derived_e0(
                repository=repository, candidate_dir=candidate_dir,
                full_e0_dir=full_e0_dir, kind=model_kind,
                model_sha256=model_sha256) != e0_binding:
            raise ValueError("Derived full E0 source lineage changed during conversion")
        if stage.is_symlink() or not stage.is_dir():
            raise RuntimeError("hls4ml did not create an ordinary project directory")
        files = sorted(stage.rglob("*"))
        if any(path.is_symlink() for path in files):
            raise ValueError("Generated hls4ml project must not contain symlinks")
        # hls4ml 1.3.0 may still emit invalid /path/type_t names in
        # generated defines.h even after its ONNX graph nodes are valid.
        # Apply a narrowly scoped, provenance-recorded C++ *identifier*
        # compatibility adaptation on disposable generated source ONLY.
        from fpgai_research.experiments.hls4ml_cpp_compat import (
            repair_unsafe_hls4ml_typedefs,
            export_failed_hls4ml_header_diagnostic,
        )
        try:
            cpp_compatibility = repair_unsafe_hls4ml_typedefs(stage)
            reject_known_invalid_hls4ml_cpp_typedefs(files)
        except ValueError:
            if failure_diagnostic_dir is not None:
                export_failed_hls4ml_header_diagnostic(
                    stage, failure_diagnostic_dir)
            raise
        generated_cpp = [p for p in files if p.is_file() and p.suffix.lower() in (".cpp", ".cc", ".cxx")]
        if not generated_cpp:
            raise ValueError("hls4ml wrote no generated HLS C++ source; refusing E1 receipt")
        config_file = stage / "hls4ml_config.yml"
        if not config_file.is_file():
            raise ValueError("hls4ml did not write its project configuration")
        # The hls4ml writer serializes its original OutputDir. After staging,
        # those internal references must point at the immutable final location.
        # Only control/config files may be rewritten; never modify HLS C++
        # computational source or generated weight bytes to fix paths.
        temporary_prefix = str(stage).encode("utf-8")
        final_prefix = str(output.absolute()).encode("utf-8")
        control_suffixes = {".yml", ".yaml", ".tcl", ".sh", ".py", ".json", ".txt", ".cfg", ".mk"}
        relocated_controls = []
        for path in files:
            if path.is_file() and path.suffix.lower() in control_suffixes:
                data = path.read_bytes()
                if temporary_prefix in data:
                    path.write_bytes(data.replace(temporary_prefix, final_prefix))
                    relocated_controls.append(path.relative_to(stage).as_posix())
        # Fail closed if an executable/configuration could still reference the
        # disappeared temporary directory, including inside generated C++.
        for path in files:
            if path.is_file() and path.stat().st_size <= 16 * 1024 * 1024:
                if temporary_prefix in path.read_bytes():
                    raise ValueError("Generated hls4ml project contains an unrelocated temporary path")
        source_inventory = [
            {"relative_path": path.relative_to(stage).as_posix(),
             "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in generated_cpp
        ]
        receipt = {
            "schema": "fpgai-research/hls4ml-baseline/v2",
            "evidence": "E1-source-only",
            "model_sha256": model_sha256,
            "model_path": str(source),
            "hls4ml_version": str(getattr(hls4ml, "__version__", "unknown")),
            "onnx_version": str(getattr(onnx, "__version__", "unknown")),
            "part": part,
            "clock_ns": clock_ns,
            "precision": precision,
            "reuse_factor": reuse_factor,
            "configuration_granularity": config_granularity,
            "name_config_policy": name_config_policy,
            "name_config_diagnostic": name_config_diagnostic,
            "requested_layer_reuse": dict(sorted((layer_reuse or {}).items())),
            "requested_single_layer_control": single_layer_intervention,
            "single_layer_control_hardware_effect_qualified": False,
            "hls4ml_layer_reuse_achieved_in_rtl": False,
            "backend": "Vitis",
            "io_type": "io_parallel",
            "hls_config": config,
            "hls_config_requested_before_conversion": requested_hls_config,
            "hls_config_changed_during_conversion":
                requested_hls_config != config,
            "generated_cpp_source_inventory": source_inventory,
            "known_invalid_cpp_typedef_guard_passed": True,
            "paper_source_onnx_inventory_reattested": inventory_sha256 is not None,
            "paper_source_onnx_inventory_sha256": inventory_sha256,
            "derived_candidate_full_e0_lineage": e0_binding,
            "hls4ml_in_memory_constant_shape_adapter": hls4ml_view,
            "hls4ml_generated_cpp_identifier_compatibility": cpp_compatibility,
            "native_e3_e4_weight_identity_verified": False,
            "relocated_control_files": relocated_controls,
            "numerical_equivalence": "NOT_VERIFIED",
            "architecture_equivalence": "NOT_ASSERTED",
            "native_synthesis": "NOT_EXECUTED",
            "note": "C++ source presence is E1 only. Verify layout, bit-level arithmetic, trained weights, tool versions and fresh native reports before comparing QoR.",
        }
        (stage / "research_baseline.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("External baseline output appeared during generation")
        stage.rename(output)
    return receipt
