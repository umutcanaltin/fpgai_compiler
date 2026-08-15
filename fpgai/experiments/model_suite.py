from __future__ import annotations

from pathlib import Path
import copy
import yaml
import torch
import torch.nn as nn


# Repository root. This keeps generated models/configs under repo-level
# models/suite and configs/suite, matching the checked-in example configs.
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "suite"
CONFIGS_DIR = ROOT / "configs" / "suite"
BENCHMARK_CONFIGS_DIR = ROOT / "examples" / "benchmark" / "models"


class MLPMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CNNMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 13 * 13, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class CNNNoPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 28 * 28, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class CNNAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 13 * 13, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class MLPSigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MLPLeakyRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CIFARSmallCNN(nn.Module):
    """Medium benchmark CNN workload with CIFAR-like 3x32x32 input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 8, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class LargeDDRStressCNN(nn.Module):
    """Larger DDR-backed stress workload for board-fit and tiling validation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class TinyYOLOLike(nn.Module):
    """Tiny detection-shaped workload: 4x4 grid with 7 values per cell."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(16 * 8 * 8, 4 * 4 * 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.head(x)
        return x.reshape(x.shape[0], 4, 4, 7)


MODEL_SPECS: list[tuple[str, nn.Module, tuple[int, ...]]] = [
    ("mlp_mnist", MLPMNIST(), (1, 1, 28, 28)),
    ("cnn_mnist", CNNMNIST(), (1, 1, 28, 28)),
    ("cnn_no_pool", CNNNoPool(), (1, 1, 28, 28)),
    ("cnn_avgpool", CNNAvgPool(), (1, 1, 28, 28)),
    ("mlp_sigmoid", MLPSigmoid(), (1, 1, 28, 28)),
    ("mlp_leakyrelu", MLPLeakyRelu(), (1, 1, 28, 28)),
    ("cifar_small_cnn", CIFARSmallCNN(), (1, 3, 32, 32)),
    ("large_ddr_stress_cnn", LargeDDRStressCNN(), (1, 3, 64, 64)),
    ("tiny_yolo_like", TinyYOLOLike(), (1, 3, 64, 64)),
]


BENCHMARK_MODEL_CONFIGS: dict[str, dict] = {
    "compact_onchip_mnist_mlp": {
        "model_name": "mlp_mnist",
        "mode": "inference",
        "weights_mode": "embedded",
        "memory_regime": "onchip",
    },
    "compact_onchip_mnist_training": {
        "model_name": "mlp_mnist",
        "mode": "training_on_device",
        "weights_mode": "embedded",
        "memory_regime": "onchip",
    },
    "medium_ddr_cifar_cnn": {
        "model_name": "cifar_small_cnn",
        "mode": "inference",
        "weights_mode": "import",
        "memory_regime": "ddr_backed",
    },
    "medium_ddr_cifar_training": {
        "model_name": "cifar_small_cnn",
        "mode": "training_on_device",
        "weights_mode": "import",
        "memory_regime": "ddr_backed",
    },
    "large_ddr_stress_cnn": {
        "model_name": "large_ddr_stress_cnn",
        "mode": "inference",
        "weights_mode": "import",
        "memory_regime": "ddr_backed_tiled",
    },
    "large_ddr_yolo_like": {
        "model_name": "tiny_yolo_like",
        "mode": "inference",
        "weights_mode": "import",
        "memory_regime": "ddr_backed_tiled",
    },
}


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def export_onnx(
    model: nn.Module,
    out_path: Path,
    input_shape: tuple[int, ...] = (1, 1, 28, 28),
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 18,
) -> None:
    model.eval()
    x = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        x,
        str(out_path),
        export_params=True,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset_version,
        dynamic_axes=None,
        dynamo=False,
    )


def make_base_config(model_rel_path: str, project_name: str) -> dict:
    return {
        "version": 1,
        "project": {
            "name": project_name,
            "out_dir": f"build/{project_name}",
            "clean": True,
        },
        "pipeline": {
            "mode": "inference",
            "outputs": {
                "top_kernel_name": "deeplearn",
            },
        },
        "targets": {
            "platform": {
                "board": "kv260",
                "part": "xck26-sfvc784-2LV-c",
                "clocks": [
                    {
                        "name": "pl_clk0",
                        "target_mhz": 200,
                    }
                ],
            }
        },
        "operators": {
            "supported": [
                "Dense",
                "Conv",
                "MaxPool",
                "AvgPool",
                "Add",
                "Relu",
                "LeakyRelu",
                "Sigmoid",
                "Softmax",
                "BatchNormalization",
                "Flatten",
                "Reshape",
            ],
            "defaults": {
                "activation_insert": {
                    "kind": "none",
                    "alpha": 0.1,
                    "except_last": True,
                }
            },
        },
        "model": {
            "format": "onnx",
            "path": model_rel_path,
        },
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
                "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
            }
        },
        "data_movement": {
            "ps_pl": {
                "compression": {
                    "enabled": True,
                },
                "weights": {
                    "mode": "embedded",
                },
            }
        },
        "backends": {
            "hls": {
                "enabled": True,
                "vitis": {
                    "enabled": True,
                    "mode": "csim",
                    "exe": "vitis_hls",
                },
            }
        },
        "toolchain": {
            "vitis_hls": {
                "enabled": True,
                "settings64": "/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh",
            },
            "vivado": {
                "enabled": True,
            },
        },
        "benchmark": {
            "enabled": True,
            "fail_on_mismatch": True,
            "seed": 0,
            "compare": {
                "atol": 0.08,
                "rtol": 0.08,
                "max_abs_error": 0.08,
                "mean_abs_error": 0.03,
                "rmse": 0.04,
                "require_argmax_match": False,
                "min_cosine_similarity": 0.95,
            },
            "intermediate": {
                "enabled": True,
                "fail_on_layer_mismatch": False,
                "stop_on_first_bad_layer": False,
            },
        },
        "debug": {
            "verbose": False,
        },
    }


def make_benchmark_config(config_name: str, spec: dict) -> dict:
    model_name = str(spec["model_name"])
    cfg = make_base_config(
        model_rel_path=f"models/suite/{model_name}.onnx",
        project_name=f"benchmark_{config_name}",
    )
    cfg["project"]["out_dir"] = f"build/benchmark/{config_name}"
    cfg["pipeline"]["mode"] = spec["mode"]
    cfg.setdefault("benchmark", {})["model_class"] = config_name
    cfg["benchmark"]["memory_regime"] = spec["memory_regime"]

    weights_mode = str(spec["weights_mode"])
    cfg["data_movement"]["ps_pl"]["weights"]["mode"] = weights_mode

    if "training" in str(spec["mode"]):
        cfg["training"] = {
            "enabled": True,
            "optimizer": {"type": "sgd", "learning_rate": 0.001},
            "loss": {"type": "mse"},
            "epochs": 1,
            "batch_size": 1,
        }

    if spec["memory_regime"] in {"ddr_backed", "ddr_backed_tiled"}:
        cfg.setdefault("optimization", {})["tiling"] = {
            "enabled": True,
            "conv": {"tm": 8, "tn": 8, "tr": 8, "tc": 8, "tk": 3},
        }
        cfg["data_movement"]["ps_pl"].setdefault("activations", {})["mode"] = "stream"
        cfg["data_movement"]["ps_pl"].setdefault("interface", {})["kind"] = "m_axi"

    return cfg



def _schema_compatible_config(cfg: dict) -> dict:
    """Normalize generated example YAMLs to the public FPGAI config schema."""
    out = copy.deepcopy(cfg)

    # Benchmark metadata belongs in benchmark_results/master_results, not as an
    # unsupported top-level compile config section.
    out.pop("benchmark", None)

    weights = (
        out.get("data_movement", {})
        .get("ps_pl", {})
        .get("weights", {})
    )
    if isinstance(weights, dict):
        mode = str(weights.get("mode", "")).strip().lower()
        aliases = {
            "m_axi": "ddr",
            "axi": "ddr",
            "axi_m": "ddr",
            "ddr_m_axi": "ddr",
            "external": "ddr",
            "external_ddr": "ddr",
            "runtime": "ddr",
            "import": "ddr",
            "import_weights": "ddr",
        }
        if mode in aliases:
            weights["mode"] = aliases[mode]

    return out

def write_yaml(cfg: dict, out_path: Path) -> None:
    cfg = _schema_compatible_config(cfg)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def create_suite() -> None:
    ensure_dirs()

    for name, model, input_shape in MODEL_SPECS:
        onnx_path = MODELS_DIR / f"{name}.onnx"
        cfg_path = CONFIGS_DIR / f"{name}.yml"

        export_onnx(model, onnx_path, input_shape=input_shape, opset_version=18)

        rel_model_path = str(onnx_path.relative_to(ROOT))
        cfg = make_base_config(
            model_rel_path=rel_model_path,
            project_name=f"fpgai_{name}",
        )
        write_yaml(cfg, cfg_path)

        print(f"[OK] model  : {onnx_path}")
        print(f"[OK] config : {cfg_path}")

    for config_name, spec in BENCHMARK_MODEL_CONFIGS.items():
        cfg = make_benchmark_config(config_name, spec)
        cfg_path = BENCHMARK_CONFIGS_DIR / f"{config_name}.yml"
        write_yaml(cfg, cfg_path)
        print(f"[OK] benchmark config : {cfg_path}")

    print()
    print("Suite generation complete.")
    print(f"Models       : {MODELS_DIR}")
    print(f"Configs      : {CONFIGS_DIR}")
    print(f"Benchmark configs: {BENCHMARK_CONFIGS_DIR}")
    print()
    print("Example run:")
    print("python -m fpgai.experiments.model_suite")


if __name__ == "__main__":
    create_suite()
