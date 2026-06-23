#!/usr/bin/env python3
"""
Sprint 16F: collect communication-aware ablation evidence from FPGAI artifacts.

This collector is conservative: it does not claim hardware runtime speedup. It scans
existing build artifacts for communication/memory plans and estimates transfer-volume
trade-offs for raw, precision-packed, and zero-run-length compressed sparse inputs.

Outputs:
  evidence/sprint16f_comm_ablation/comm_ablation.csv
  evidence/sprint16f_comm_ablation/comm_ablation.json
  evidence/sprint16f_comm_ablation/comm_ablation.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class Row:
    experiment: str
    design: str
    mode: str
    precision: str
    input_elements: int | None
    raw_bytes: int | None
    transfer_bytes: int | None
    dma_words_32b: int | None
    compression_ratio_vs_raw: float | None
    assumed_sparsity: float | None
    pack_time_us: str
    latency_cycles: str
    lut: str
    ff: str
    bram: str
    dsp: str
    evidence_source: str
    notes: str


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def walk_values(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_values(v)


def find_first_int(obj: Any, names: set[str]) -> int | None:
    for k, v in walk_values(obj):
        if str(k).lower() in names:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
            if isinstance(v, str):
                try:
                    x = int(float(v))
                    if x > 0:
                        return x
                except Exception:
                    pass
    return None


def find_precision(artifact: Path) -> str:
    # Prefer obvious design-name precision.
    name = artifact.name.lower()
    if 'fx8' in name:
        return 'fx8_like'
    if 'fx12' in name:
        return 'fx12_like'
    if 'fx16' in name:
        return 'fx16_like'

    for p in [
        artifact / 'build' / 'ir' / 'layerwise_precision.json',
        artifact / 'build' / 'hls' / 'codegen_meta.json',
        artifact / 'build' / 'manifest.json',
    ]:
        obj = load_json(p)
        if obj is None:
            continue
        text = json.dumps(obj).lower()
        if 'fx8' in text or 'ap_fixed<8' in text:
            return 'fx8_like'
        if 'fx12' in text or 'ap_fixed<12' in text:
            return 'fx12_like'
        if 'fx16' in text or 'ap_fixed<16' in text:
            return 'fx16_like'
        if 'ap_fixed' in text:
            return 'fixed_point'
    return 'unknown'


def infer_input_elements(artifact: Path) -> tuple[int | None, str, str]:
    candidates = [
        artifact / 'build' / 'ir' / 'comm_plan.json',
        artifact / 'build' / 'ir' / 'memory_plan.json',
        artifact / 'build' / 'ir' / 'descriptors.json',
        artifact / 'build' / 'manifest.json',
        artifact / 'build' / 'hls_artifact_metadata.json',
    ]
    names = {
        'input_elements', 'num_input_elements', 'n_input_elements', 'input_size',
        'input_count', 'num_inputs', 'input_len', 'input_length', 'activation_elements',
        'total_input_elements', 'input_words'
    }
    for p in candidates:
        obj = load_json(p)
        if obj is None:
            continue
        val = find_first_int(obj, names)
        if val is not None:
            return val, str(p), 'found_explicit_input_element_count'

    # Fallback for MNIST-like experiments used in the current FPGAI evidence.
    text = artifact.name.lower()
    if 'mnist' in text or 'cnn' in text or 'training' in text or 'hw_' in text:
        return 28 * 28, 'fallback:mnist_28x28', 'fallback_assumption_mnist_input_784'

    return None, '', 'missing_input_size'


def parse_vivado_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        design = r.get('design') or ''
        if design:
            out[design] = r
    return out


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_rows(experiment: Path, vivado: dict[str, dict[str, str]], sparsity: float) -> list[Row]:
    rows: list[Row] = []
    artifacts = experiment / 'artifacts'
    if not artifacts.exists():
        return rows

    for artifact in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        design = artifact.name
        precision = find_precision(artifact)
        input_elements, source, note = infer_input_elements(artifact)
        if input_elements is None:
            rows.append(Row(str(experiment), design, 'unknown', precision, None, None, None, None, None, None, '', '', '', '', '', '', source, note))
            continue

        raw_bytes = input_elements * 4

        # Conservative modeled modes. These are transfer-volume ablations, not board runtime.
        modes: list[tuple[str, int, float | None, str]] = []
        modes.append(('raw_fp32', raw_bytes, 1.0, 'baseline_32bit_float_transfer'))

        # 8/12/16-bit payloads are rounded to full bytes per element.
        if precision == 'fx8_like':
            packed_bytes = input_elements * 1
        elif precision == 'fx12_like':
            packed_bytes = ceil_div(input_elements * 12, 8)
        elif precision == 'fx16_like' or precision == 'fixed_point':
            packed_bytes = input_elements * 2
        else:
            packed_bytes = input_elements * 2
        modes.append(('precision_packed', packed_bytes, raw_bytes / packed_bytes if packed_bytes else None, 'modeled_precision_packing'))

        # Simple zero-run-length sparse encoding model: nonzero value + run/count token.
        nonzero = max(1, int(round(input_elements * (1.0 - sparsity))))
        zrl_bytes = nonzero * 2 + nonzero * 2
        modes.append(('zero_run_length_sparse', zrl_bytes, raw_bytes / zrl_bytes if zrl_bytes else None, f'modeled_zrl_sparse_input_sparsity_{sparsity:g}'))

        vr = vivado.get(design, {})
        for mode, transfer_bytes, ratio, mode_note in modes:
            rows.append(Row(
                experiment=str(experiment),
                design=design,
                mode=mode,
                precision=precision,
                input_elements=input_elements,
                raw_bytes=raw_bytes,
                transfer_bytes=transfer_bytes,
                dma_words_32b=ceil_div(transfer_bytes, 4),
                compression_ratio_vs_raw=round(ratio, 4) if ratio is not None else None,
                assumed_sparsity=sparsity if 'zero_run_length' in mode else None,
                pack_time_us='',
                latency_cycles='',
                lut=vr.get('lut', '') or vr.get('LUT', ''),
                ff=vr.get('ff', '') or vr.get('FF', ''),
                bram=vr.get('bram', '') or vr.get('BRAM', ''),
                dsp=vr.get('dsp', '') or vr.get('DSP', ''),
                evidence_source=source,
                notes=f'{note};{mode_note}',
            ))
    return rows


def fmt(v: Any) -> str:
    if v is None:
        return ''
    return str(v)


def write_outputs(rows: list[Row], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'comm_ablation.csv'
    json_path = out / 'comm_ablation.json'
    md_path = out / 'comm_ablation.md'

    fieldnames = list(asdict(rows[0]).keys()) if rows else [f.name for f in Row.__dataclass_fields__.values()]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding='utf-8')

    designs = sorted({r.design for r in rows})
    reduced = [r for r in rows if r.mode != 'raw_fp32' and r.transfer_bytes is not None and r.raw_bytes is not None and r.transfer_bytes < r.raw_bytes]

    with md_path.open('w', encoding='utf-8') as f:
        f.write('# Sprint 16F Communication-Aware Ablation Evidence\n\n')
        f.write('This table summarizes transfer-volume ablations derived from existing FPGAI communication/memory artifacts. It does not claim physical-board runtime improvement.\n\n')
        f.write('## Summary\n\n')
        f.write(f'- design count: {len(designs)}\n')
        f.write(f'- rows: {len(rows)}\n')
        f.write(f'- rows with reduced transfer volume vs raw: {len(reduced)}\n\n')
        f.write('| design | mode | precision | input_elements | raw_bytes | transfer_bytes | dma_words_32b | compression_ratio_vs_raw | assumed_sparsity | notes |\n')
        f.write('|---|---|---|---:|---:|---:|---:|---:|---:|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join([
                r.design, r.mode, r.precision, fmt(r.input_elements), fmt(r.raw_bytes), fmt(r.transfer_bytes), fmt(r.dma_words_32b), fmt(r.compression_ratio_vs_raw), fmt(r.assumed_sparsity), r.notes
            ]) + ' |\n')
        f.write('\n## Safe claim\n\n')
        f.write('FPGAI communication artifacts can be used to quantify transfer-volume trade-offs between raw, precision-packed, and sparse zero-run-length-style transfer formats for evaluated designs.\n\n')
        f.write('## Limitation\n\n')
        f.write('This sprint reports transfer-volume evidence only. It does not measure physical-board DMA latency or runtime speedup. Physical runtime validation is reserved for later board-validation sprints.\n')

    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')
    print(f'rows={len(rows)} designs={len(designs)} reduced_rows={len(reduced)}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('experiments', nargs='+', help='Experiment directories to scan')
    ap.add_argument('--vivado-summary', default='', help='Optional Sprint 16C Vivado summary CSV')
    ap.add_argument('--out', default='evidence/sprint16f_comm_ablation')
    ap.add_argument('--sparsity', type=float, default=0.90, help='Assumed sparsity for modeled zero-run-length sparse input')
    args = ap.parse_args()

    vivado = parse_vivado_summary(Path(args.vivado_summary)) if args.vivado_summary else {}
    rows: list[Row] = []
    for exp in args.experiments:
        rows.extend(build_rows(Path(exp), vivado, args.sparsity))

    write_outputs(rows, Path(args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
