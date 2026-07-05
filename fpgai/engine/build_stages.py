from __future__ import annotations

from typing import Any, Dict

from fpgai.config.access import get_path


BUILD_STAGE_KEYS = (
    "cpp",
    "testbench",
    "hls_project",
    "hls_synthesis",
    "vivado_project",
    "vivado_implementation",
    "bitstream",
    "runtime_package",
    "reports",
    "host_cpp",
)


def cfg_has_path(raw: Dict[str, Any], path: str) -> bool:
    cur: Any = raw
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def resolve_build_stages(raw: Dict[str, Any]) -> Dict[str, bool]:
    """Resolve user-facing build.stages into an explicit stage contract.

    Legacy configs without build.stages keep the old behavior: HLS source/project
    generation follows backends.hls.enabled, Vitis execution follows
    toolchain.vitis_hls.enabled, host C++ follows backends.host_cpp.enabled, and
    runtime package/report metadata are emitted.
    """
    explicit = cfg_has_path(raw, "build.stages")
    requested = get_path(raw, "build.stages", {}) or {}
    if requested is not None and not isinstance(requested, dict):
        raise ValueError("build.stages must be a mapping of stage names to booleans.")

    legacy_hls = as_bool(get_path(raw, "backends.hls.enabled", True))
    legacy_host = as_bool(get_path(raw, "backends.host_cpp.enabled", True))
    legacy_hls_run = as_bool(get_path(raw, "toolchain.vitis_hls.enabled", False))

    if explicit:
        stages: Dict[str, bool] = {
            "cpp": True,
            "testbench": True,
            "hls_project": False,
            "hls_synthesis": False,
            "vivado_project": False,
            "vivado_implementation": False,
            "bitstream": False,
            "runtime_package": True,
            "reports": True,
            "host_cpp": legacy_host,
        }
        unknown = sorted(set(requested) - set(BUILD_STAGE_KEYS) - {"existing_hls_ip"})
        if unknown:
            raise ValueError(
                "Unsupported build.stages keys: " + ", ".join(unknown) + ". "
                "Supported keys are: " + ", ".join(BUILD_STAGE_KEYS) + "."
            )
        for key, value in requested.items():
            if key in stages:
                stages[key] = as_bool(value)
    else:
        stages = {
            "cpp": legacy_hls,
            "testbench": legacy_hls,
            "hls_project": legacy_hls,
            "hls_synthesis": legacy_hls and legacy_hls_run,
            "vivado_project": False,
            "vivado_implementation": False,
            "bitstream": False,
            "runtime_package": True,
            "reports": True,
            "host_cpp": legacy_host,
        }

    validate_build_stage_dependencies(raw, stages)
    return stages


def validate_build_stage_dependencies(raw: Dict[str, Any], stages: Dict[str, bool]) -> None:
    if stages.get("testbench") and not stages.get("cpp"):
        raise ValueError("build.stages.testbench=true requires build.stages.cpp=true.")
    if stages.get("hls_project") and not stages.get("cpp"):
        raise ValueError("build.stages.hls_project=true requires build.stages.cpp=true.")
    if stages.get("hls_synthesis") and not stages.get("hls_project"):
        raise ValueError("build.stages.hls_synthesis=true requires build.stages.hls_project=true.")

    existing_hls_ip = as_bool(get_path(raw, "build.existing_hls_ip", False))
    if stages.get("vivado_project") and not (stages.get("hls_synthesis") or existing_hls_ip):
        raise ValueError(
            "build.stages.vivado_project=true requires build.stages.hls_synthesis=true "
            "or build.existing_hls_ip=true."
        )
    if stages.get("vivado_implementation") and not stages.get("vivado_project"):
        raise ValueError("build.stages.vivado_implementation=true requires build.stages.vivado_project=true.")
    if stages.get("bitstream") and not (stages.get("vivado_project") and stages.get("vivado_implementation")):
        raise ValueError(
            "build.stages.bitstream=true requires build.stages.vivado_project=true "
            "and build.stages.vivado_implementation=true."
        )


def build_stage_summary(stages: Dict[str, bool]) -> Dict[str, Any]:
    return {key: bool(stages.get(key, False)) for key in BUILD_STAGE_KEYS}
