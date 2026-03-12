from __future__ import annotations

from pathlib import Path
import yaml
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "suite"
CONFIGS_DIR = ROOT / "configs" / "suite"


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


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


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

    # Force the classic exporter path to avoid noisy exporter/version-converter issues
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


def write_yaml(cfg: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def create_suite() -> None:
    ensure_dirs()

    suite = [
        ("mlp_mnist", MLPMNIST()),
        ("cnn_mnist", CNNMNIST()),
        ("cnn_no_pool", CNNNoPool()),
        ("cnn_avgpool", CNNAvgPool()),
        ("mlp_sigmoid", MLPSigmoid()),
        ("mlp_leakyrelu", MLPLeakyRelu()),
    ]

    for name, model in suite:
        onnx_path = MODELS_DIR / f"{name}.onnx"
        cfg_path = CONFIGS_DIR / f"{name}.yml"

        export_onnx(model, onnx_path, opset_version=18)

        rel_model_path = str(onnx_path.relative_to(ROOT))
        cfg = make_base_config(
            model_rel_path=rel_model_path,
            project_name=f"fpgai_{name}",
        )
        write_yaml(cfg, cfg_path)

        print(f"[OK] model  : {onnx_path}")
        print(f"[OK] config : {cfg_path}")

    print()
    print("Suite generation complete.")
    print(f"Models : {MODELS_DIR}")
    print(f"Configs: {CONFIGS_DIR}")
    print()
    print("Example run:")
    print("python3 main.py --config configs/suite/cnn_mnist.yml")


if __name__ == "__main__":
    create_suite()