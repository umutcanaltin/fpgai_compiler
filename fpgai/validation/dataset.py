from __future__ import annotations

import csv
import importlib
import json
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


def emit_dataset_artifacts(
    out_dir: str | Path,
    *,
    raw_config: dict[str, Any] | None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    raw = raw_config or {}
    validation = raw.get("validation", {}) if isinstance(raw, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    dataset_cfg = validation.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return DatasetArtifacts(
            status="not_configured", source="not_provided", sample_count=0, input_shape=(),
            inputs_path=None, labels_path=None, targets_path=None, manifest_path=None,
            summary_path=None, reason="validation.dataset is not configured",
        ).as_dict()

    out = Path(out_dir)
    root = out / "validation" / "dataset"
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
