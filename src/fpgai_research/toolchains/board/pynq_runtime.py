"""PYNQ-Z2 Linux-side prototype for an FPGAI 32-bit AXI DMA overlay.

Run only ON an actual, compatible PYNQ-Z2 board/image after validating the
generated bitstream + HWH and explicitly authorizing programming.
One 32-bit transport word per signed fixed-point tensor element, lower W
bits, AP_RND/AP_SAT numerical contract; no runtime weight updates.
No PYNQ dependency is imported when this module is inspected on a PC.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import sys
import time


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def encode_ap_fixed(values, *, width: int, integer_bits: int):
    """Pack host float32 to the low W bits of 32-bit AXI transport words."""
    import numpy as np
    if not (2 <= width <= 32 and 1 <= integer_bits <= width):
        raise ValueError("Invalid fixed-point type")
    array = np.asarray(values)
    if array.dtype != np.float32 or not np.all(np.isfinite(array)):
        raise ValueError("Input must be a finite original float32 tensor")
    fraction = width - integer_bits
    # AP_RND rounds to positive infinity on an exact half-step;
    # saturation and two's complement are explicit.
    scaled = np.floor(array.astype(np.float64) * (2.0 ** fraction) + 0.5)
    minimum = -(2 ** (width - 1))
    maximum = (2 ** (width - 1)) - 1
    signed = np.clip(scaled, minimum, maximum).astype(np.int64)
    return np.asarray(signed & ((1 << width) - 1), dtype=np.uint32)


def decode_ap_fixed(words, *, width: int, integer_bits: int):
    import numpy as np
    if not (2 <= width <= 32 and 1 <= integer_bits <= width):
        raise ValueError("Invalid fixed-point type")
    raw = np.asarray(words)
    if raw.dtype != np.uint32:
        raise ValueError("AXI DMA receive words must be uint32")
    data = raw.astype(np.int64) & ((1 << width) - 1)
    signed = (data ^ (1 << (width - 1))) - (1 << (width - 1))
    return (signed.astype(np.float64) / (2.0 ** (width - integer_bits))).astype(np.float32)


def _verify_bundle(summary_path: Path, bit: Path,
                   *, allow_advisories: bool = False) -> dict:
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError("Missing or symlinked original board build summary")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or summary.get("schema") != "fpgai-research/board-dma-build/v1":
        raise ValueError("Unsupported board build receipt")
    clean = (summary.get("state") == "BOARD_BITSTREAM_DRC_CLEAN_UNTESTED_ON_DEVICE"
             and summary.get("strict_board_drc_clean_qualified") is True
             and summary.get("drc_clean") is True)
    warning_only = (
        summary.get("state") == "BOARD_BITSTREAM_WITH_ADVISORIES_NOT_DEVICE_TESTED"
        and summary.get("warning_only_eligible") is True
        and summary.get("user_accepted_board_advisories") is True
        and summary.get("drc_clean") is False
        and summary.get("drc_critical_warnings") == 0
        and summary.get("drc_unconstrained_io") is False
        and allow_advisories is True
    )
    if summary.get("board") != "pynq-z2" or not (clean or warning_only):
        raise ValueError("Refusing to program an unqualified board design or unapproved DRC advisories")
    if bit.is_symlink() or not bit.is_file() or not bit.name.endswith(".bit"):
        raise ValueError("Missing/symlinked .bit image")
    hwh = bit.with_suffix(".hwh")
    if hwh.is_symlink() or not hwh.is_file():
        raise ValueError("Matching .hwh metadata must accompany the .bit")
    if (_sha(bit) != summary.get("bitstream_sha256")
            or _sha(hwh) != summary.get("hwh_sha256")):
        raise ValueError("Bitstream/HWH bytes differ from original board build receipt")
    input_shape = summary.get("input_elements")
    output_shape = summary.get("output_elements")
    w = summary.get("payload_bits")
    i = summary.get("integer_bits")
    if (any(type(v) is not int for v in (input_shape, output_shape, w, i))
            or not (1 <= input_shape <= 262144 and 1 <= output_shape <= 262144)
            or not (2 <= w <= 32 and 1 <= i <= w)):
        raise ValueError("Invalid source-pinned fixed transport geometry")
    in_shape = summary.get("input_shape")
    out_shape = summary.get("output_shape")
    if (not isinstance(in_shape, list) or not isinstance(out_shape, list)
            or not in_shape or not out_shape
            or any(type(d) is not int or d < 1 for d in in_shape + out_shape)
            or prod(in_shape) != input_shape or prod(out_shape) != output_shape):
        raise ValueError("Shape and element count differ in source-pinned board manifest")
    return summary


def _wait_dma_polling(dma, timeout_s: float) -> None:
    """Avoid indefinite DMA.wait() if TLAST never arrives or hardware hangs."""
    deadline = time.monotonic() + timeout_s
    while not (dma.sendchannel.idle and dma.recvchannel.idle):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "AXI DMA did not complete; check TLAST, clock/reset and interrupt "
                "configuration. Hardware may need reset before another attempt."
            )
        time.sleep(0.002)
    dma.sendchannel.wait()
    dma.recvchannel.wait()


def execute_on_board(*, summary_path: str, bit_path: str,
                     input_npy: str, output_npy: str,
                     expected_npy: str | None,
                     tolerance: float, timeout_seconds: float,
                     confirm_board: bool,
                     accept_board_advisories: bool = False) -> dict:
    import numpy as np

    if confirm_board is not True:
        raise ValueError("Explicit --confirm-pynq-z2 is required before programming the FPGA")
    if not 0.0 < tolerance <= 10.0 or not 0.1 <= timeout_seconds <= 600:
        raise ValueError("Expected tolerance 0..10 and DMA timeout 0.1..600 seconds")
    raw_bit = Path(bit_path).expanduser()
    if raw_bit.is_symlink():
        raise ValueError("Symlinked bitstream rejected")
    bit = raw_bit.resolve(strict=True)
    summary = _verify_bundle(
        Path(summary_path).expanduser(), bit,
        allow_advisories=accept_board_advisories,
    )
    inputs = np.load(Path(input_npy).expanduser(), allow_pickle=False)
    if inputs.dtype != np.float32 or tuple(inputs.shape) != tuple(summary["input_shape"]):
        raise ValueError("Input must be float32 with the exact compiler tensor shape")
    expected = (np.load(Path(expected_npy).expanduser(), allow_pickle=False)
                if expected_npy else None)
    if expected is not None and (
        expected.dtype != np.float32
        or tuple(expected.shape) != tuple(summary["output_shape"])
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError("Expected output must be finite float32 of exact model output length")
    destination = Path(output_npy).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite on-device inference output")
    payload = encode_ap_fixed(inputs, width=summary["payload_bits"],
                              integer_bits=summary["integer_bits"])

    try:
        from pynq import Overlay, allocate
    except ImportError as exc:
        raise RuntimeError("This runtime must execute on a PYNQ-Z2 image with the pynq package") from exc

    overlay = Overlay(str(bit), download=True)
    if not {"axi_dma_0", "fpgai_dma_0"} <= set(overlay.ip_dict):
        raise ValueError("PYNQ HWH does not describe the expected DMA and accelerator")
    dma = overlay.axi_dma_0
    accelerator = overlay.fpgai_dma_0
    src = allocate(shape=(summary["input_elements"],), dtype=np.uint32)
    dst = allocate(shape=(summary["output_elements"],), dtype=np.uint32)
    try:
        src[:] = payload.reshape(-1)
        dst[:] = 0
        src.sync_to_device()
        dst.sync_to_device()
        start = time.monotonic()
        dma.recvchannel.transfer(dst)
        dma.sendchannel.transfer(src)
        accelerator.write(0x00, 0x01)  # HLS ap_ctrl_hs AP_START
        _wait_dma_polling(dma, timeout_seconds)
        elapsed = time.monotonic() - start
        dst.sync_from_device()
        decoded = decode_ap_fixed(
            np.asarray(dst), width=summary["payload_bits"],
            integer_bits=summary["integer_bits"],
        ).reshape(tuple(summary["output_shape"]))
    finally:
        src.freebuffer()
        dst.freebuffer()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as out:
        np.save(out, decoded, allow_pickle=False)

    result = {
        "schema": "fpgai-research/pynq-z2-on-board-inference/v1",
        "board": "pynq-z2",
        "bitstream_sha256": _sha(bit),
        "inference_output_npy": str(destination),
        "hardware_transfer_completed": True,
        "dma_and_control_elapsed_s": elapsed,
        "expected_output_supplied": expected is not None,
        "board_drc_clean": summary.get("drc_clean"),
        "warning_only_board_programming_explicitly_accepted": (
            summary.get("drc_clean") is False and accept_board_advisories),
        "numerical_verification_status": "NOT_PROVIDED",
        "on_device_numerics_verified": False,
        "note": "Elapsed time includes DMA/control overhead; no accelerator-only latency claim.",
    }
    if expected is not None:
        diff = np.abs(decoded.astype(np.float64) - expected.reshape(-1))
        result["max_absolute_error"] = float(diff.max())
        result["rmse"] = float(np.sqrt(np.mean(diff * diff)))
        result["numerical_verification_status"] = (
            "PASS" if np.all(np.isfinite(diff)) and float(diff.max()) <= tolerance
            else "FAIL"
        )
        result["on_device_numerics_verified"] = result["numerical_verification_status"] == "PASS"
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-summary", required=True)
    p.add_argument("--bit", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected", help="Independent software reference .npy, float32")
    p.add_argument("--tolerance", type=float, default=.1)
    p.add_argument("--timeout-seconds", type=float, default=30)
    p.add_argument("--confirm-pynq-z2", action="store_true")
    p.add_argument("--accept-board-advisories", action="store_true",
                   help="Explicitly allow programming a warning-only board image "
                        "accepted in the source-pinned build receipt")
    args = p.parse_args(argv)
    try:
        receipt = execute_on_board(
            summary_path=args.build_summary, bit_path=args.bit,
            input_npy=args.input, output_npy=args.output,
            expected_npy=args.expected, tolerance=args.tolerance,
            timeout_seconds=args.timeout_seconds,
            confirm_board=args.confirm_pynq_z2,
            accept_board_advisories=args.accept_board_advisories,
        )
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        print(f"FPGAI PYNQ runtime failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["on_device_numerics_verified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
