from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _find_designs(root: Path):
    art = root / 'artifacts'
    if not art.exists():
        return []
    return sorted([p for p in art.iterdir() if p.is_dir()])


def _rel_paths(base: Path, name: str):
    return [str(p.relative_to(base)) for p in base.rglob(name)] if base.exists() else []


def _read_text(p: Path) -> str:
    try:
        return p.read_text(errors='ignore')
    except Exception:
        return ''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m fpgai.reporting.training_native_accumulated_batch_artifacts <experiment_dir>')
    root = Path(sys.argv[1])
    rows = []
    for d in _find_designs(root):
        build = d / 'build'
        manifest = _load_json(build / 'manifest.json') or {}
        status = ''
        try:
            data = _load_json(root / 'results.json') or {}
            for rec in data.get('records', data if isinstance(data, list) else []):
                if isinstance(rec, dict) and rec.get('design_name') == d.name:
                    status = rec.get('status', '')
                    break
        except Exception:
            pass
        compare = manifest.get('training_compare') or {}
        summary = None
        for p in build.rglob('training_multistep_summary.json'):
            summary = _load_json(p)
            if summary:
                break
        summary = summary or {}
        hls_cpp = _read_text(build / 'hls' / 'src' / 'deeplearn.cpp')
        native_markers = all(x in hls_cpp for x in [
            'FPGAI_NATIVE_ACC_BATCH_COUNT',
            'if (mode == 3)',
            'if (mode == 4)',
            'if (mode == 5)',
        ])
        rows.append({
            'design': d.name,
            'status': status or manifest.get('status', ''),
            'weights_mode': 'stream' if '_stream_' in d.name or d.name.startswith('training_cnn_stream') else 'embedded',
            'hls_ok': bool(_rel_paths(build, 'weights_before.bin') and _rel_paths(build, 'grads.bin') and _rel_paths(build, 'weights_after.bin')),
            'training_compare': bool(compare),
            'native_accumulated_optimizer': native_markers,
            'accumulated_batch': bool(summary.get('accumulated_batch')),
            'averaged_gradients': bool(summary.get('averaged_gradients')),
            'gradient_accumulation_mode': bool(summary.get('gradient_accumulation_mode')) or ('if (mode == 3)' in hls_cpp),
            'optimizer_apply_mode': bool(summary.get('optimizer_apply_mode')) or ('if (mode == 4)' in hls_cpp),
            'reset_accumulator_mode': bool(summary.get('reset_accumulator_mode')) or ('if (mode == 5)' in hls_cpp),
            'optimizer_location': summary.get('optimizer_location', ''),
            'train_steps': summary.get('train_steps', ''),
            'batch_size': summary.get('batch_size', ''),
            'total_forward_backward_calls': summary.get('total_forward_backward_calls', summary.get('total_train_calls', '')),
            'optimizer_update_calls': summary.get('optimizer_update_calls', ''),
            'weight_words': summary.get('weight_words', ''),
            'grad_cosine': compare.get('grad_cosine', ''),
            'weight_after_cosine': compare.get('weight_after_cosine', ''),
            'weight_delta_cosine': compare.get('weight_delta_cosine', ''),
        })
    cols = [
        'design','status','weights_mode','hls_ok','training_compare',
        'native_accumulated_optimizer','accumulated_batch','averaged_gradients',
        'gradient_accumulation_mode','optimizer_apply_mode','reset_accumulator_mode',
        'optimizer_location','train_steps','batch_size','total_forward_backward_calls',
        'optimizer_update_calls','weight_words','grad_cosine','weight_after_cosine','weight_delta_cosine'
    ]
    out_dir = root / 'training_native_accumulated_batch_artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'artifacts.json').write_text(json.dumps(rows, indent=2))
    md = ['# Native accumulated mini-batch training artifacts', '', '| ' + ' | '.join(cols) + ' |', '|'+ '|'.join(['---']*len(cols))+'|']
    for r in rows:
        md.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    text = '\n'.join(md) + '\n'
    (out_dir / 'artifacts.md').write_text(text)
    print(text)
    print(f'[OK] Wrote {out_dir}')


if __name__ == '__main__':
    main()
