#!/usr/bin/env python3
"""
Collect reviewer-safe claim support for FPGAI.

This script creates:
  reports/reproducibility/claim_support.csv
  reports/reproducibility/claim_support.md

Rule:
  Do not invent claims. A claim is marked READY only if at least one
  supporting artifact path exists in the repository.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Claim:
    claim: str
    safe_wording: str
    evidence_paths: tuple[str, ...]
    reproduce_command: str
    paper_section: str
    limitations: str


CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim="Evaluated generated in-PL training convergence smoke tests",
        safe_wording=(
            "FPGAI supports evaluated small multi-epoch HLS CSim convergence "
            "smoke tests for generated CNN training accelerators using native "
            "accumulated mini-batch SGD."
        ),
        evidence_paths=(
            "experiments/sprint13f_training_multi_epoch_convergence_v3",
            "experiments/sprint13f_training_multi_epoch_convergence_v2_smoke",
            "experiments/sprint13f_training_multi_epoch_convergence_smoke",
        ),
        reproduce_command=(
            "fpgai report build "
            "experiments/sprint13f_training_multi_epoch_convergence_v3 "
            "--out reports/training_convergence"
        ),
        paper_section="Training correctness and convergence",
        limitations=(
            "Small convergence smoke test only; not validation for arbitrary ONNX "
            "training or large-model FPGA training."
        ),
    ),
    Claim(
        claim="Native accumulated mini-batch SGD training flow",
        safe_wording=(
            "FPGAI-generated training designs include evaluated forward, backward, "
            "gradient accumulation, and SGD-update behavior for the tested training "
            "accelerator configurations."
        ),
        evidence_paths=(
            "experiments/sprint13e_training_native_accumulated_batch_v3",
            "experiments/sprint13e_training_native_accumulated_batch_v3_smoke",
            "experiments/sprint13d_training_accumulated_batch",
        ),
        reproduce_command=(
            "fpgai experiment run "
            "--sweep configs/sweeps/sprint13e_training_native_accumulated_batch.yml "
            "--out experiments/sprint13e_training_native_accumulated_batch_v3"
        ),
        paper_section="In-PL training architecture",
        limitations=(
            "Validated only for the evaluated generated training accelerators and "
            "available operator subset."
        ),
    ),
    Claim(
        claim="Hardware knob validation for precision",
        safe_wording=(
            "FPGAI hardware precision settings produce measurable implementation-level "
            "differences in Vivado-reported power, timing slack, LUT, FF, BRAM, and DSP "
            "usage for evaluated designs."
        ),
        evidence_paths=(
            "experiments/sprint15a_hardware_knob_validation_parallel",
            "experiments/sprint15a_hardware_knob_validation_smoke",
        ),
        reproduce_command=(
            "fpgai experiment run "
            "--sweep configs/sweeps/sprint15a_hardware_knob_validation.yml "
            "--out experiments/sprint15a_hardware_knob_validation_parallel"
        ),
        paper_section="Hardware knob validation",
        limitations=(
            "Validation is limited to evaluated precision modes and target device flow."
        ),
    ),
    Claim(
        claim="Pipeline policy changes implementation results",
        safe_wording=(
            "FPGAI pipeline policy settings produce measurable generated-code and "
            "Vivado implementation-level differences for evaluated designs."
        ),
        evidence_paths=(
            "experiments/sprint15c_pipeline_policy_strengthened",
            "experiments/sprint15b_pipeline_only",
        ),
        reproduce_command=(
            "fpgai experiment run "
            "--sweep configs/sweeps/sprint15c_pipeline_policy_strengthened.yml "
            "--out experiments/sprint15c_pipeline_policy_strengthened"
        ),
        paper_section="Pipeline policy evaluation",
        limitations=(
            "Pipeline policy is not a global optimizer and does not guarantee optimal II."
        ),
    ),
    Claim(
        claim="Parallel feasibility envelope",
        safe_wording=(
            "FPGAI identifies evaluated parallel design points as pass, timing_fail, "
            "or resource_fail using implementation-level timing and resource reports."
        ),
        evidence_paths=(
            "experiments/sprint15e_parallel_feasible_envelope",
        ),
        reproduce_command=(
            "fpgai experiment run "
            "--sweep configs/sweeps/sprint15e_parallel_feasible_envelope.yml "
            "--out experiments/sprint15e_parallel_feasible_envelope"
        ),
        paper_section="Feasibility and resource envelope",
        limitations=(
            "Classification is based on evaluated design points; FPGAI does not prove "
            "global optimality across the full design space."
        ),
    ),
    Claim(
        claim="Implementation feasibility reporting",
        safe_wording=(
            "FPGAI reports implementation feasibility, resource utilization, timing "
            "slack, estimated Fmax, and recommended safe-clock settings for evaluated "
            "designs."
        ),
        evidence_paths=(
            "experiments/sprint15e_parallel_feasible_envelope",
            "experiments/sprint15c_pipeline_policy_strengthened",
            "experiments/sprint15a_hardware_knob_validation_parallel",
        ),
        reproduce_command=(
            "fpgai report build "
            "--out reports/reproducibility"
        ),
        paper_section="Hardware feasibility and safe-clock reporting",
        limitations=(
            "Current release reports safe-clock guidance but does not automatically "
            "modify user clock constraints."
        ),
    ),
)


def existing_paths(root: Path, paths: Iterable[str]) -> list[str]:
    found: list[str] = []
    for p in paths:
        if (root / p).exists():
            found.append(p)
    return found


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--out", default="reports/reproducibility", help="Output directory")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for c in CLAIMS:
        found = existing_paths(repo, c.evidence_paths)
        status = "READY" if found else "MISSING"
        rows.append(
            {
                "claim": c.claim,
                "safe_wording": c.safe_wording,
                "artifact_folder": "; ".join(found) if found else "",
                "candidate_artifact_paths": "; ".join(c.evidence_paths),
                "reproduce_command": c.reproduce_command,
                "paper_section": c.paper_section,
                "status": status,
                "limitations": c.limitations,
            }
        )

    csv_path = out / "claim_support.csv"
    md_path = out / "claim_support.md"

    fieldnames = [
        "claim",
        "safe_wording",
        "artifact_folder",
        "candidate_artifact_paths",
        "reproduce_command",
        "paper_section",
        "status",
        "limitations",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# FPGAI Claim Support Matrix\n\n")
        f.write(
            "This table maps reviewer-facing paper claims to available repository "
            "artifacts. Claims marked `READY` have at least one existing supporting "
            "artifact path. Claims marked `MISSING` must not be used in the paper yet.\n\n"
        )
        f.write("| Claim | Safe wording | Artifact folder | Reproduce command | Paper section | Status | Limitations |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    md_escape(r[k])
                    for k in [
                        "claim",
                        "safe_wording",
                        "artifact_folder",
                        "reproduce_command",
                        "paper_section",
                        "status",
                        "limitations",
                    ]
                )
                + " |\n"
            )

    ready = sum(1 for r in rows if r["status"] == "READY")
    missing = sum(1 for r in rows if r["status"] == "MISSING")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"READY={ready} MISSING={missing}")

    if missing:
        print("Missing claims exist. Do not use MISSING claims in the paper.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
