"""Fail-fast source-binding check for licensed ORIGINAL Vitis HLS campaigns.

Pytest config prepends src/, but `python -m rl_environment.gym_curriculum`
does not do so by itself. A stale non-editable fpgai_research installation
can therefore pass CPU tests and fail at the first native packaging call.
Stop BEFORE creating a native study or spending any vendor invocation.
"""
from __future__ import annotations

from inspect import signature
from pathlib import Path


def verify_local_compiler_package() -> dict:
    from fpgai_research.toolchains.vitis_hls import package as pkg
    from . import post_hls_env

    expected = (
        Path(__file__).resolve().parents[1] / "src" / "fpgai_research"
        / "toolchains" / "vitis_hls" / "package.py"
    ).resolve(strict=True)
    # Do not require a stale imported module's recorded path to exist: some
    # old editable/zip installs leave relocated or removed __file__ paths.
    actual = Path(pkg.__file__).resolve()
    hint = (
        "Native FPGAI research imported an outdated/different installed "
        "fpgai_research compiler, not this checkout's src tree. "
        "From the repository root with the intended virtual environment run "
        "'python -m pip install -e .' (or set PYTHONPATH=$PWD/src:$PWD), "
        "then retry in a NEW experiment output directory; "
        "do not overwrite a partially created original study."
    )
    if actual != expected:
        raise RuntimeError(
            f"{hint} Imported package: {actual}; checkout package: {expected}.")
    for name in ("make_vitis_tcl", "write_vitis_package"):
        params = signature(getattr(pkg, name)).parameters
        if "clock_uncertainty_ns" not in params:
            raise RuntimeError(
                f"{hint} Local {name} lacks clock_uncertainty_ns.")
    # The backend's from-import must reference precisely the checked code.
    if post_hls_env.write_vitis_package is not pkg.write_vitis_package:
        raise RuntimeError(
            f"{hint} Native Gym packaging binding does not match inspected source.")
    return {
        "status": "LOCAL_CURRENT_SOURCE_VITIS_PACKAGING_BINDING_VERIFIED",
        "resolved_compiler_package": str(actual),
        "source_package_matches_checkout": True,
        "clock_uncertainty_keyword_supported": True,
        "original_vitis_calls": 0,
    }


def main():
    import json
    print(json.dumps(verify_local_compiler_package(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
