from __future__ import annotations

import csv
import importlib
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetArtifacts:
    status: str
    source: str
    sample_count: int
    input_shape: tuple[int, ...]
    inputs_path: Path | None
    labels_path: Path | None
    targets_path: Path | None
    manifest_path: Path | None
    summary_path: Path | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "sample_count": self.sample_count,
            "input_shape": list(self.input_shape),
            "input_words_per_sample": int(np.prod(self.input_shape)) if self.input_shape else 0,
            "inputs_path": self.inputs_path,
            "labels_path": self.labels_path,
            "targets_path": self.targets_path,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "reason": self.reason,
        }


def _resolve(path_value: Any, *, base_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _selection_indices(
    dataset_cfg: dict[str, Any],
    *,
    count: int,
    labels: np.ndarray | None,
) -> np.ndarray:
    cfg = dataset_cfg.get("sample_selection", {})
    if not isinstance(cfg, dict):
        cfg = {}
    mode = str(cfg.get("mode") or "first").strip().lower().replace("-", "_")
    offset = max(0, int(cfg.get("offset", 0) or 0))
    requested = int(cfg.get("count", count - offset) or (count - offset))
    requested = max(0, min(requested, count - offset))
    seed = int(cfg.get("seed", 0) or 0)
    candidates = np.arange(offset, count, dtype=np.int64)
    if requested <= 0:
        return candidates[:0]
    if mode == "first":
        return candidates[:requested]
    rng = np.random.default_rng(seed)
    if mode == "random":
        return np.sort(rng.choice(candidates, size=requested, replace=False))
    if mode in {"balanced", "balanced_per_class"}:
        if labels is None:
            raise ValueError("balanced sample selection requires classification labels")
        flat_labels = np.asarray(labels).reshape(-1)
        if flat_labels.size != count:
            raise ValueError(f"label count {flat_labels.size} does not match dataset sample count {count}")
        classes = sorted(int(v) for v in np.unique(flat_labels[candidates]))
        if not classes:
            raise ValueError("balanced sample selection found no classes")
        per_class = cfg.get("per_class_count")
        if per_class is None:
            per_class = requested // len(classes)
        per_class = int(per_class)
        if per_class <= 0:
            raise ValueError("balanced sample selection requires count >= number of classes or per_class_count > 0")
        selected: list[int] = []
        for class_id in classes:
            class_candidates = candidates[flat_labels[candidates] == class_id]
            if class_candidates.size < per_class:
                raise ValueError(
                    f"class {class_id} has {class_candidates.size} samples, fewer than requested {per_class}"
                )
            chosen = rng.choice(class_candidates, size=per_class, replace=False)
            selected.extend(int(v) for v in chosen)
        remaining = requested - len(selected)
        if remaining > 0:
            selected_set = set(selected)
            pool = np.asarray([int(v) for v in candidates if int(v) not in selected_set], dtype=np.int64)
            if pool.size < remaining:
                raise ValueError("balanced sample selection cannot satisfy requested count")
            selected.extend(int(v) for v in rng.choice(pool, size=remaining, replace=False))
        return np.asarray(sorted(selected[:requested]), dtype=np.int64)
    raise ValueError(f"unsupported validation.dataset.sample_selection.mode: {mode!r}")


def _load_torchvision_source(
    dataset_cfg: dict[str, Any],
    *,
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, None, str, Path, dict[str, Any]]:
    name = str(dataset_cfg.get("name") or "").strip()
    if name not in {"MNIST", "FashionMNIST"}:
        raise ValueError("validation.dataset.name must be one of ['FashionMNIST', 'MNIST']")
    split = str(dataset_cfg.get("split") or "test").strip().lower()
    if split not in {"train", "test"}:
        raise ValueError("validation.dataset.split must be one of ['test', 'train']")
    root = _resolve(dataset_cfg.get("root") or "datasets", base_dir=base_dir)
    assert root is not None
    download = bool(dataset_cfg.get("download", False))
    try:
        torchvision = importlib.import_module("torchvision")
    except Exception as exc:
        raise RuntimeError(
            "torchvision dataset source requires the optional 'datasets' dependencies "
            "(torch and torchvision)"
        ) from exc
    dataset_cls = getattr(torchvision.datasets, name, None)
    if dataset_cls is None:
        raise RuntimeError(f"installed torchvision does not provide dataset {name}")
    try:
        dataset = dataset_cls(root=str(root), train=(split == "train"), download=download)
    except Exception as exc:
        action = "enable download: true" if not download else "check dataset/network availability"
        raise RuntimeError(f"could not load torchvision {name} {split} split; {action}: {exc}") from exc
    data = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets, dtype=np.int64).reshape(-1)
    if data.shape[0] != labels.size:
        raise ValueError(f"torchvision data count {data.shape[0]} does not match label count {labels.size}")
    preprocessing = dataset_cfg.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        preprocessing = {}
    inputs = data.astype(np.float32, copy=False)
    normalize = bool(preprocessing.get("normalize", True))
    if normalize:
        inputs = inputs / 255.0
    mean = preprocessing.get("mean")
    std = preprocessing.get("std")
    if mean is not None or std is not None:
        mean_value = float(0.0 if mean is None else mean)
        std_value = float(1.0 if std is None else std)
        if std_value == 0.0:
            raise ValueError("validation.dataset.preprocessing.std must be non-zero")
        inputs = (inputs - mean_value) / std_value
    if bool(preprocessing.get("add_channel_dim", False)) and inputs.ndim == 3:
        inputs = inputs[:, None, :, :]
    if bool(preprocessing.get("flatten", True)):
        inputs = inputs.reshape((inputs.shape[0], -1))
    class_names = list(getattr(dataset, "classes", [str(i) for i in range(10)]))
    provenance = {
        "dataset_name": name,
        "split": split,
        "root": str(root),
        "download": download,
        "class_names": class_names,
        "torchvision_version": str(getattr(torchvision, "__version__", "unknown")),
        "preprocessing": {
            "normalize": normalize,
            "flatten": bool(preprocessing.get("flatten", True)),
            "add_channel_dim": bool(preprocessing.get("add_channel_dim", False)),
            "mean": mean,
            "std": std,
        },
    }
    return inputs, labels, None, "torchvision", root, provenance

def _load_source(dataset_cfg: dict[str, Any], *, base_dir: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str, Path, dict[str, Any]]:
    source = str(dataset_cfg.get("source") or "").strip().lower().replace("-", "_")
    if source == "torchvision":
        return _load_torchvision_source(dataset_cfg, base_dir=base_dir)
    path = _resolve(dataset_cfg.get("path") or dataset_cfg.get("inputs"), base_dir=base_dir)
    if path is None:
        raise ValueError("validation.dataset.path is required")
    if not path.exists():
        raise FileNotFoundError(path)
    if not source:
        source = path.suffix.lower().lstrip(".")

    labels: np.ndarray | None = None
    targets: np.ndarray | None = None
    if source == "npy":
        inputs = np.load(path, allow_pickle=False)
    elif source == "npz":
        archive = np.load(path, allow_pickle=False)
        inputs_key = str(dataset_cfg.get("inputs_key") or "inputs")
        if inputs_key not in archive:
            if len(archive.files) == 1:
                inputs_key = archive.files[0]
            else:
                raise KeyError(f"NPZ inputs key {inputs_key!r} not found; available keys: {archive.files}")
        inputs = archive[inputs_key]
        labels_key = str(dataset_cfg.get("labels_key") or "labels")
        targets_key = str(dataset_cfg.get("targets_key") or "targets")
        if labels_key in archive:
            labels = np.asarray(archive[labels_key])
        if targets_key in archive:
            targets = np.asarray(archive[targets_key])
    elif source in {"binary", "bin", "raw_binary"}:
        dtype = np.dtype(str(dataset_cfg.get("dtype") or "float32"))
        raw = np.fromfile(path, dtype=dtype)
        sample_shape = tuple(int(v) for v in (dataset_cfg.get("sample_shape") or []))
        if not sample_shape:
            raise ValueError("validation.dataset.sample_shape is required for binary input")
        words = int(np.prod(sample_shape))
        if words <= 0 or raw.size % words != 0:
            raise ValueError(f"binary input word count {raw.size} is not divisible by sample size {words}")
        inputs = raw.reshape((-1, *sample_shape))
        labels_path = _resolve(dataset_cfg.get("labels") or dataset_cfg.get("labels_path"), base_dir=base_dir)
        targets_path = _resolve(dataset_cfg.get("targets") or dataset_cfg.get("targets_path"), base_dir=base_dir)
        if labels_path is not None:
            labels = np.fromfile(labels_path, dtype=np.dtype(str(dataset_cfg.get("labels_dtype") or "int64")))
        if targets_path is not None:
            targets = np.fromfile(targets_path, dtype=np.dtype(str(dataset_cfg.get("targets_dtype") or "float32")))
    else:
        raise ValueError(f"unsupported validation.dataset.source: {source!r}")

    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim == 0:
        raise ValueError("dataset inputs must contain a sample dimension")
    if inputs.ndim == 1:
        inputs = inputs.reshape((1, -1))
    return inputs, labels, targets, source, path, {}



def _tensor_word_count(shape: Any) -> int:
    """Return the static number of scalar values represented by a tensor shape."""
    dims = tuple(int(value) for value in (shape or ()))
    if not dims:
        return 0
    count = 1
    for value in dims:
        if value <= 0:
            return 0
        count *= value
    return int(count)


def emit_dataset_model_contract(
    out_dir: str | Path,
    *,
    graph: Any,
    dataset_artifacts: dict[str, Any],
    require_supervision: bool = False,
) -> dict[str, Any]:
    """Validate normalized dataset records against the imported FPGAI graph.

    The generated HLS testbench consumes flattened records. Therefore semantic
    compatibility requires equal scalar word counts; the original logical shapes
    are still recorded so future layout-aware lowering can strengthen this gate.
    The function emits its report before callers raise on incompatibility, which
    keeps failed experiment points diagnosable and traceable.
    """
    reports_dir = Path(out_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "training_dataset_model_contract.json"
    markdown_path = reports_dir / "training_dataset_model_contract.md"

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    dataset_status = str(dataset_artifacts.get("status") or "unknown")
    if dataset_status != "available":
        payload = {
            "artifact_kind": "fpgai_training_dataset_model_contract",
            "schema_version": 1,
            "status": "not_available",
            "reason": str(dataset_artifacts.get("reason") or "dataset artifacts are unavailable"),
            "checks": checks,
            "warnings": warnings,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(
            "# Training dataset/model contract\n\n"
            f"- Status: `not_available`\n"
            f"- Reason: {payload['reason']}\n",
            encoding="utf-8",
        )
        return payload | {"json_path": json_path, "markdown_path": markdown_path}

    graph_inputs = list(getattr(graph, "inputs", []) or [])
    graph_outputs = list(getattr(graph, "outputs", []) or [])
    add_check(
        "single_model_input",
        len(graph_inputs) == 1,
        f"model input count={len(graph_inputs)}; dataset training currently requires exactly one input",
    )
    add_check(
        "single_model_output",
        len(graph_outputs) == 1,
        f"model output count={len(graph_outputs)}; dataset training currently requires exactly one output",
    )

    input_name = graph_inputs[0] if len(graph_inputs) == 1 else None
    output_name = graph_outputs[0] if len(graph_outputs) == 1 else None
    input_spec = graph.get_tensor(input_name) if input_name is not None else None
    output_spec = graph.get_tensor(output_name) if output_name is not None else None
    model_input_shape = tuple(int(v) for v in (getattr(input_spec, "shape", None) or ()))
    model_output_shape = tuple(int(v) for v in (getattr(output_spec, "shape", None) or ()))
    model_input_words = _tensor_word_count(model_input_shape)
    model_output_words = _tensor_word_count(model_output_shape)

    dataset_input_shape = tuple(int(v) for v in (dataset_artifacts.get("input_shape") or ()))
    dataset_input_words = int(
        dataset_artifacts.get("input_words_per_sample")
        or _tensor_word_count(dataset_input_shape)
    )
    input_compatible = model_input_words > 0 and dataset_input_words == model_input_words
    add_check(
        "input_word_count",
        input_compatible,
        (
            f"dataset sample shape={list(dataset_input_shape)} ({dataset_input_words} words); "
            f"model input {input_name!r} shape={list(model_input_shape)} ({model_input_words} words)"
        ),
    )

    sample_count = int(dataset_artifacts.get("sample_count") or 0)
    inputs_path_value = dataset_artifacts.get("inputs_path")
    input_stats: dict[str, Any] = {}
    if inputs_path_value:
        inputs_path = Path(str(inputs_path_value))
        if inputs_path.exists():
            inputs = np.asarray(np.load(inputs_path, allow_pickle=False), dtype=np.float32)
            flat = inputs.reshape(-1)
            input_stats = {
                "minimum": float(np.min(flat)) if flat.size else None,
                "maximum": float(np.max(flat)) if flat.size else None,
                "mean": float(np.mean(flat)) if flat.size else None,
                "standard_deviation": float(np.std(flat)) if flat.size else None,
                "nonzero_fraction": float(np.count_nonzero(flat) / flat.size) if flat.size else 0.0,
            }
            if flat.size and float(np.std(flat)) == 0.0:
                warnings.append("dataset inputs have zero variance")
            if flat.size and int(np.count_nonzero(flat)) == 0:
                warnings.append("dataset inputs are all zero")

    labels_path_value = dataset_artifacts.get("labels_path")
    targets_path_value = dataset_artifacts.get("targets_path")
    labels_summary: dict[str, Any] = {"status": "not_provided"}
    targets_summary: dict[str, Any] = {"status": "not_provided"}
    supervision_available = False

    if labels_path_value:
        labels_path = Path(str(labels_path_value))
        if labels_path.exists():
            labels = np.asarray(np.load(labels_path, allow_pickle=False), dtype=np.int64).reshape(-1)
            classes, counts = np.unique(labels, return_counts=True)
            labels_summary = {
                "status": "provided",
                "count": int(labels.size),
                "minimum": int(labels.min()) if labels.size else None,
                "maximum": int(labels.max()) if labels.size else None,
                "class_count": int(classes.size),
                "class_distribution": {str(int(k)): int(v) for k, v in zip(classes, counts)},
            }
            supervision_available = labels.size == sample_count
            add_check(
                "label_count",
                labels.size == sample_count,
                f"label count={labels.size}; selected sample count={sample_count}",
            )
            label_range_ok = (
                labels.size > 0
                and model_output_words > 1
                and bool(np.all(labels >= 0))
                and bool(np.all(labels < model_output_words))
            )
            add_check(
                "label_range",
                label_range_ok,
                (
                    f"label range={labels_summary['minimum']}..{labels_summary['maximum']}; "
                    f"model output width={model_output_words}"
                ),
            )
            if classes.size <= 1:
                warnings.append("classification labels contain only one class")

    if targets_path_value:
        targets_path = Path(str(targets_path_value))
        if targets_path.exists():
            targets = np.asarray(np.load(targets_path, allow_pickle=False), dtype=np.float32)
            target_words = int(np.prod(targets.shape[1:])) if targets.ndim > 1 else 1
            targets_summary = {
                "status": "provided",
                "shape": [int(v) for v in targets.shape],
                "target_words_per_sample": target_words,
            }
            target_count_ok = targets.ndim > 0 and int(targets.shape[0]) == sample_count
            target_width_ok = target_words == model_output_words
            add_check(
                "target_count",
                target_count_ok,
                f"target record count={int(targets.shape[0]) if targets.ndim else 0}; selected sample count={sample_count}",
            )
            add_check(
                "target_word_count",
                target_width_ok,
                f"target words per sample={target_words}; model output width={model_output_words}",
            )
            supervision_available = target_count_ok and target_width_ok

    if require_supervision:
        add_check(
            "supervision_available",
            supervision_available,
            "training datasets require labels or targets compatible with the model output",
        )

    passed = all(bool(item["passed"]) for item in checks)
    if "dataset inputs are all zero" in warnings or "classification labels contain only one class" in warnings:
        claim_scope = "mechanism_smoke_only"
    elif sample_count < 100:
        claim_scope = "small_sample_learning_smoke_only"
    else:
        claim_scope = "learning_evaluation_candidate_requires_held_out_validation"

    failed_details = [str(item["detail"]) for item in checks if not bool(item["passed"])]
    reason = (
        "Dataset records are compatible with the imported FPGAI graph."
        if passed
        else "Dataset/model compatibility failed: " + "; ".join(failed_details)
    )
    payload = {
        "artifact_kind": "fpgai_training_dataset_model_contract",
        "schema_version": 1,
        "status": "compatible" if passed else "incompatible",
        "reason": reason,
        "model": {
            "input_name": input_name,
            "input_shape": list(model_input_shape),
            "input_words": model_input_words,
            "output_name": output_name,
            "output_shape": list(model_output_shape),
            "output_words": model_output_words,
        },
        "dataset": {
            "sample_count": sample_count,
            "input_shape_per_sample": list(dataset_input_shape),
            "input_words_per_sample": dataset_input_words,
            "source": str(dataset_artifacts.get("source") or "unknown"),
            "input_statistics": input_stats,
            "labels": labels_summary,
            "targets": targets_summary,
        },
        "compatibility_basis": "flattened_scalar_word_count",
        "claim_scope": claim_scope,
        "checks": checks,
        "warnings": warnings,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Training dataset/model contract",
        "",
        f"- Status: `{payload['status']}`",
        f"- Reason: {reason}",
        f"- Compatibility basis: `{payload['compatibility_basis']}`",
        f"- Claim scope: `{claim_scope}`",
        f"- Model input: `{input_name}` `{list(model_input_shape)}` / `{model_input_words}` words",
        f"- Dataset sample: `{list(dataset_input_shape)}` / `{dataset_input_words}` words",
        f"- Model output: `{output_name}` `{list(model_output_shape)}` / `{model_output_words}` words",
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        marker = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{marker}` `{item['name']}` — {item['detail']}")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return payload | {"json_path": json_path, "markdown_path": markdown_path}

def emit_dataset_artifacts(
    out_dir: str | Path,
    *,
    raw_config: dict[str, Any] | None,
    base_dir: str | Path | None = None,
    dataset_config: dict[str, Any] | None = None,
    artifact_subdir: str = "dataset",
) -> dict[str, Any]:
    raw = raw_config or {}
    validation = raw.get("validation", {}) if isinstance(raw, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    dataset_cfg = dataset_config if isinstance(dataset_config, dict) else validation.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return DatasetArtifacts(
            status="not_configured", source="not_provided", sample_count=0, input_shape=(),
            inputs_path=None, labels_path=None, targets_path=None, manifest_path=None,
            summary_path=None, reason="validation.dataset is not configured",
        ).as_dict()

    out = Path(out_dir)
    root = out / "validation" / str(artifact_subdir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        inputs, labels, targets, source, source_path, provenance = _load_source(
            dataset_cfg, base_dir=Path(base_dir or Path.cwd())
        )
        source_count = int(inputs.shape[0])
        indices = _selection_indices(dataset_cfg, count=source_count, labels=labels)
        count = int(indices.size)
        if count <= 0:
            raise ValueError("dataset sample selection produced zero samples")
        inputs = np.ascontiguousarray(inputs[indices], dtype=np.float32)
        if labels is not None:
            labels = np.asarray(labels).reshape(-1)[indices].astype(np.int64, copy=False)
            if labels.size != count:
                raise ValueError(f"label count {labels.size} does not match selected sample count {count}")
        if targets is not None:
            targets = np.asarray(targets)[indices].astype(np.float32, copy=False)
            if targets.shape[0] != count:
                raise ValueError(f"target count {targets.shape[0]} does not match selected sample count {count}")

        inputs_path = root / "inputs.npy"
        inputs_bin = root / "inputs.bin"
        np.save(inputs_path, inputs)
        inputs.reshape(-1).tofile(inputs_bin)
        labels_path: Path | None = None
        targets_path: Path | None = None
        if labels is not None:
            labels_path = root / "labels.npy"
            np.save(labels_path, labels)
        if targets is not None:
            targets_path = root / "targets.npy"
            np.save(targets_path, targets)

        sample_index = root / "sample_index.csv"
        with sample_index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["normalized_index", "source_index"])
            for index in range(count):
                writer.writerow([index, int(indices[index])])

        manifest = {
            "artifact_kind": "fpgai_dataset_manifest",
            "schema_version": 1,
            "status": "available",
            "source": source,
            "source_path": str(source_path),
            "sample_count": count,
            "sample_offset": int(indices.min()) if indices.size else 0,
            "sample_indices": [int(v) for v in indices],
            "input_shape_per_sample": list(inputs.shape[1:]),
            "input_dtype": "float32",
            "input_words_per_sample": int(np.prod(inputs.shape[1:])),
            "input_words_total": int(inputs.size),
            "labels_status": "provided" if labels is not None else "not_provided",
            "targets_status": "provided" if targets is not None else "not_provided",
            "selection": dataset_cfg.get("sample_selection", {}) if isinstance(dataset_cfg.get("sample_selection"), dict) else {},
            "class_distribution": ({str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))} if labels is not None else {}),
            "provenance": provenance,
            "artifacts": {
                "inputs_npy": str(inputs_path),
                "inputs_bin": str(inputs_bin),
                "labels": str(labels_path) if labels_path else None,
                "targets": str(targets_path) if targets_path else None,
                "sample_index_csv": str(sample_index),
            },
        }
        manifest_path = root / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary_path = root / "dataset_summary.md"
        summary_path.write_text(
            "\n".join([
                "# Dataset normalization summary", "",
                f"- Status: `available`", f"- Source: `{source}`",
                f"- Samples: `{count}`", f"- Input shape per sample: `{list(inputs.shape[1:])}`",
                f"- Labels: `{manifest['labels_status']}`", f"- Targets: `{manifest['targets_status']}`", "",
            ]), encoding="utf-8"
        )
        return DatasetArtifacts(
            status="available", source=source, sample_count=count,
            input_shape=tuple(int(v) for v in inputs.shape[1:]), inputs_path=inputs_path,
            labels_path=labels_path, targets_path=targets_path, manifest_path=manifest_path,
            summary_path=summary_path, reason="dataset normalized into compiler validation artifacts",
        ).as_dict() | {"inputs_bin": inputs_bin, "inputs_array": inputs}
    except Exception as exc:
        manifest_path = root / "dataset_manifest.json"
        payload = {
            "artifact_kind": "fpgai_dataset_manifest", "schema_version": 1,
            "status": "invalid", "reason": str(exc),
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return DatasetArtifacts(
            status="invalid", source=str(dataset_cfg.get("source") or "unknown"), sample_count=0,
            input_shape=(), inputs_path=None, labels_path=None, targets_path=None,
            manifest_path=manifest_path, summary_path=None, reason=str(exc),
        ).as_dict()

def emit_training_validation_dataset_artifacts(
    out_dir: str | Path,
    *,
    raw_config: dict[str, Any] | None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize an optional held-out dataset without creating a second loader.

    Configuration owner: ``validation.training_validation.dataset``.
    The returned artifact schema matches :func:`emit_dataset_artifacts`, while
    files are written below ``validation/held_out_dataset``.
    """
    raw = raw_config or {}
    validation = raw.get("validation", {}) if isinstance(raw, dict) else {}
    training_validation = validation.get("training_validation", {}) if isinstance(validation, dict) else {}
    dataset_cfg = training_validation.get("dataset") if isinstance(training_validation, dict) else None
    if not isinstance(dataset_cfg, dict):
        return DatasetArtifacts(
            status="not_configured", source="not_provided", sample_count=0, input_shape=(),
            inputs_path=None, labels_path=None, targets_path=None, manifest_path=None,
            summary_path=None, reason="validation.training_validation.dataset is not configured",
        ).as_dict()
    return emit_dataset_artifacts(
        out_dir, raw_config=raw, base_dir=base_dir, dataset_config=dataset_cfg,
        artifact_subdir="held_out_dataset",
    )


def _manifest_digest(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit_training_validation_split_contract(
    out_dir: str | Path,
    *,
    training_artifacts: dict[str, Any],
    validation_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Publish a deterministic train/held-out split contract.

    Non-overlap is checked from source identity plus selected source indices.
    Different source files are treated as independently owned splits.
    """
    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    def load_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
        value = artifacts.get("manifest_path")
        path = Path(value) if value else None
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    train_manifest = load_manifest(training_artifacts)
    valid_manifest = load_manifest(validation_artifacts)
    train_status = str(training_artifacts.get("status") or "not_configured")
    valid_status = str(validation_artifacts.get("status") or "not_configured")

    train_source = str(train_manifest.get("source_path") or "")
    valid_source = str(valid_manifest.get("source_path") or "")
    train_indices = {int(v) for v in train_manifest.get("sample_indices", [])}
    valid_indices = {int(v) for v in valid_manifest.get("sample_indices", [])}
    same_source = bool(train_source and valid_source and Path(train_source).resolve() == Path(valid_source).resolve())
    overlap = sorted(train_indices.intersection(valid_indices)) if same_source else []

    if train_status != "available":
        status = "not_available"
        reason = "training dataset is not available"
    elif valid_status != "available":
        status = "not_configured" if valid_status == "not_configured" else "invalid"
        reason = str(validation_artifacts.get("reason") or "held-out dataset is not available")
    elif overlap:
        status = "incompatible"
        reason = f"training and held-out selections overlap at {len(overlap)} source record(s)"
    else:
        status = "compatible"
        reason = "training and held-out dataset selections are disjoint"

    payload = {
        "artifact_kind": "fpgai_training_validation_split_contract",
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "claim_scope": "held_out_validation_mechanism" if status == "compatible" else "not_available",
        "training": {
            "sample_count": int(training_artifacts.get("sample_count") or 0),
            "source_path": train_source or None,
            "sample_indices": sorted(train_indices),
            "class_distribution": train_manifest.get("class_distribution", {}),
            "manifest_sha256": _manifest_digest(Path(training_artifacts["manifest_path"])) if training_artifacts.get("manifest_path") else None,
        },
        "validation": {
            "sample_count": int(validation_artifacts.get("sample_count") or 0),
            "source_path": valid_source or None,
            "sample_indices": sorted(valid_indices),
            "class_distribution": valid_manifest.get("class_distribution", {}),
            "manifest_sha256": _manifest_digest(Path(validation_artifacts["manifest_path"])) if validation_artifacts.get("manifest_path") else None,
        },
        "same_source": same_source,
        "overlap_count": len(overlap),
        "overlap_indices": overlap,
        "statistical_generalization_claim": False,
    }
    json_path = reports / "training_validation_split_contract.json"
    md_path = reports / "training_validation_split_contract.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join([
        "# FPGAI training/held-out validation split contract", "",
        f"- Status: `{status}`",
        f"- Training samples: `{payload['training']['sample_count']}`",
        f"- Held-out samples: `{payload['validation']['sample_count']}`",
        f"- Same source: `{same_source}`",
        f"- Overlap count: `{len(overlap)}`",
        f"- Claim scope: `{payload['claim_scope']}`",
        "- Statistical generalization claim: `False`", "",
    ]), encoding="utf-8")
    return payload | {"json_path": str(json_path), "markdown_path": str(md_path)}
