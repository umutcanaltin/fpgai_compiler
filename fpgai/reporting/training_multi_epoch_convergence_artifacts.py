from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


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


def _find_paths(base: Path, name: str):
    return list(base.rglob(name)) if base.exists() else []


def _read_text(p: Path) -> str:
    try:
        return p.read_text(errors='ignore')
    except Exception:
        return ''


def _status(root: Path, design: str) -> str:
    data = _load_json(root / 'results.json')
    if isinstance(data, dict):
        records = data.get('records', [])
    elif isinstance(data, list):
        records = data
    else:
        records = []
    for rec in records:
        if isinstance(rec, dict) and rec.get('design_name') == design:
            return str(rec.get('status', ''))
    return ''


def _float_or_blank(x: Any):
    if x in (None, ''):
        return ''
    try:
        return float(x)
    except Exception:
        return ''


def _int_or_blank(x: Any):
    if x in (None, ''):
        return ''
    try:
        return int(x)
    except Exception:
        return ''


def _bin_word_count(paths: list[Path]) -> int | str:
    if not paths:
        return ''
    try:
        return paths[0].stat().st_size // 4
    except Exception:
        return ''


def _collect_logs(build: Path) -> str:
    candidates = [
        build / 'hls' / 'logs' / 'vitis_hls_stdout.log',
        build / 'hls' / 'vitis_hls.log',
        build / 'hls' / 'fpgai_hls_proj' / 'sol1' / 'csim' / 'report' / 'deeplearn_csim.log',
    ]
    parts = []
    for p in candidates:
        t = _read_text(p)
        if t:
            parts.append(t)
    return '\n'.join(parts)


def _summary_from_logs(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r'convergence_initial_loss=([-+0-9.eE]+)\s+loss_eval_records=(\d+)', log_text)
    if m:
        out['initial_loss'] = float(m.group(1))
        out['loss_eval_records'] = int(m.group(2))
    epoch_matches = re.findall(r'convergence_epoch=(\d+)\s+loss=([-+0-9.eE]+)', log_text)
    if epoch_matches:
        epoch_losses = [float(v) for _, v in sorted(((int(e), v) for e, v in epoch_matches), key=lambda x: x[0])]
        out['epoch_losses'] = epoch_losses
        out['final_loss'] = epoch_losses[-1]
        if 'initial_loss' in out:
            out['loss_delta'] = out['initial_loss'] - out['final_loss']
            out['loss_decreased'] = out['final_loss'] < out['initial_loss']
            out['multi_epoch_convergence'] = bool(len(epoch_losses) >= 1 and out['final_loss'] <= out['initial_loss'])
    m = re.search(r'train_steps=(\d+)\s+batch_size=(\d+)\s+in_words=(\d+)\s+out_words=(\d+)', log_text)
    if m:
        out['train_steps'] = int(m.group(1))
        out['batch_size'] = int(m.group(2))
    m = re.search(r'Multi-step summary:\s*train_steps=(\d+)\s+batch_size=(\d+)\s+total_train_calls=(\d+)', log_text)
    if m:
        out['train_steps'] = int(m.group(1))
        out['batch_size'] = int(m.group(2))
        out['total_forward_backward_calls'] = int(m.group(3))
    if 'native_accumulated_batch=true optimizer_location=hls_top_accumulated_optimizer' in log_text:
        out['optimizer_location'] = 'hls_top_accumulated_optimizer'
    return out


def _merge_summary(json_summary: dict[str, Any], log_summary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(json_summary or {})
    for k, v in log_summary.items():
        if merged.get(k, '') in ('', None, [], False):
            merged[k] = v
    return merged


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m fpgai.reporting.training_multi_epoch_convergence_artifacts <experiment_dir>')
    root = Path(sys.argv[1])
    rows = []
    for d in _find_designs(root):
        build = d / 'build'
        manifest = _load_json(build / 'manifest.json') or {}
        compare = manifest.get('training_compare') or {}

        summary = None
        for name in ('training_multiepoch_summary.json', 'training_multi_epoch_summary.json', 'training_multistep_summary.json'):
            for p in build.rglob(name):
                summary = _load_json(p)
                if summary:
                    break
            if summary:
                break
        summary = summary or {}
        log_text = _collect_logs(build)
        summary = _merge_summary(summary, _summary_from_logs(log_text))

        hls_cpp = _read_text(build / 'hls' / 'src' / 'deeplearn.cpp')
        native_markers = all(x in hls_cpp for x in [
            'FPGAI_NATIVE_ACC_BATCH_COUNT',
            'if (mode == 3)',
            'if (mode == 4)',
            'if (mode == 5)',
        ])
        loss_eval_mode = ('if (mode == 6)' in hls_cpp) or bool(summary.get('loss_eval_mode'))
        hls_ok = bool(_rel_paths(build, 'weights_before.bin') and _rel_paths(build, 'grads.bin') and _rel_paths(build, 'weights_after.bin'))
        initial = _float_or_blank(summary.get('initial_loss'))
        final = _float_or_blank(summary.get('final_loss'))
        if summary.get('loss_delta', '') not in ('', None):
            loss_delta = _float_or_blank(summary.get('loss_delta'))
        elif initial != '' and final != '':
            loss_delta = initial - final
        else:
            loss_delta = ''
        epoch_losses = summary.get('epoch_losses', [])
        if isinstance(epoch_losses, list):
            epoch_losses_str = '[' + ', '.join(str(x) for x in epoch_losses) + ']'
            epoch_count = len(epoch_losses)
        else:
            epoch_losses_str = str(epoch_losses)
            epoch_count = ''
        train_steps = _int_or_blank(summary.get('train_steps'))
        batch_size = _int_or_blank(summary.get('batch_size'))
        total_fwbw = _int_or_blank(summary.get('total_forward_backward_calls', summary.get('total_train_calls')))
        if total_fwbw == '' and train_steps != '' and batch_size != '':
            total_fwbw = train_steps * batch_size
        optimizer_updates = _int_or_blank(summary.get('optimizer_update_calls'))
        if optimizer_updates == '' and train_steps != '':
            optimizer_updates = train_steps
        optimizer_location = summary.get('optimizer_location', '')
        if not optimizer_location and native_markers:
            optimizer_location = 'hls_top_accumulated_optimizer'
        weight_words = summary.get('weight_words', '')
        if weight_words in ('', None):
            weight_words = _bin_word_count(_find_paths(build, 'weights_before.bin'))
        loss_decreased = bool(summary.get('loss_decreased'))
        if initial != '' and final != '':
            loss_decreased = final < initial
        multi_epoch = bool(summary.get('multi_epoch_convergence'))
        if hls_ok and loss_eval_mode and initial != '' and final != '' and epoch_count:
            multi_epoch = final <= initial

        rows.append({
            'design': d.name,
            'status': _status(root, d.name) or manifest.get('status', ''),
            'weights_mode': 'stream' if '_stream_' in d.name or d.name.startswith('training_cnn_stream') else 'embedded',
            'hls_ok': hls_ok,
            'training_compare': bool(compare),
            'native_accumulated_optimizer': native_markers,
            'loss_eval_mode': loss_eval_mode,
            'multi_epoch_convergence': multi_epoch,
            'loss_decreased': loss_decreased,
            'initial_loss': initial,
            'final_loss': final,
            'loss_delta': loss_delta,
            'epoch_losses': epoch_losses_str,
            'epoch_loss_count': epoch_count,
            'train_steps': train_steps,
            'batch_size': batch_size,
            'loss_eval_records': summary.get('loss_eval_records', ''),
            'total_forward_backward_calls': total_fwbw,
            'optimizer_update_calls': optimizer_updates,
            'optimizer_location': optimizer_location,
            'weight_words': weight_words,
            'grad_cosine': compare.get('grad_cosine', ''),
            'weight_after_cosine': compare.get('weight_after_cosine', ''),
            'weight_delta_cosine': compare.get('weight_delta_cosine', ''),
        })
    cols = [
        'design','status','weights_mode','hls_ok','training_compare',
        'native_accumulated_optimizer','loss_eval_mode','multi_epoch_convergence','loss_decreased',
        'initial_loss','final_loss','loss_delta','epoch_losses','epoch_loss_count',
        'train_steps','batch_size','loss_eval_records','total_forward_backward_calls',
        'optimizer_update_calls','optimizer_location','weight_words','grad_cosine',
        'weight_after_cosine','weight_delta_cosine'
    ]
    out_dir = root / 'training_multi_epoch_convergence_artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'artifacts.json').write_text(json.dumps(rows, indent=2))
    md = ['# Multi-epoch convergence artifacts', '', '| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for r in rows:
        md.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    text = '\n'.join(md) + '\n'
    (out_dir / 'artifacts.md').write_text(text)
    print(text)
    print(f'[OK] Wrote {out_dir}')


if __name__ == '__main__':
    main()
