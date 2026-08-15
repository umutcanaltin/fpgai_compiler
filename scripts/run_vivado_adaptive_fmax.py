from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fpgai.analysis.board_clock_planner import (
    realizable_targets_from_probes,
    run_clock_probe,
    write_clock_probe_report,
)


def main() -> None:
    ap=argparse.ArgumentParser(description="Probe realizable board clocks, deduplicate them, then run an implementation Fmax sweep")
    ap.add_argument('--config',required=True)
    ap.add_argument('--board',default='kv260')
    ap.add_argument('--probe-requests',nargs='+',type=float,required=True)
    ap.add_argument('--out',default='build/adaptive_fmax')
    ap.add_argument('--vivado',default='vivado')
    ap.add_argument('--continue-on-failure',action='store_true')
    args=ap.parse_args()

    root=Path(args.out).resolve(); root.mkdir(parents=True,exist_ok=True)
    probe_root=root/'clock_probe'; probe_root.mkdir(parents=True,exist_ok=True)
    points=[]
    for mhz in args.probe_requests:
        point=run_clock_probe(board_name=args.board,requested_mhz=mhz,out_dir=probe_root/f'{mhz:g}mhz',vivado_executable=args.vivado)
        points.append(point)
        print(f'[CLOCK-PROBE] {mhz:g} MHz -> {point.actual_mhz if point.actual_mhz is not None else "unresolved"} MHz')
    probe_report=write_clock_probe_report(points,probe_root/'board_clock_probe.json')
    targets=realizable_targets_from_probes(points)
    plan={
        'schema':'fpgai.adaptive-fmax-plan/v1',
        'board':args.board,
        'probe_report':str(probe_report),
        'requested_probe_count':len(args.probe_requests),
        'unique_realizable_targets_mhz':list(targets),
        'deduplicated_implementation_count':len(targets),
        'measurement_semantics':'Only unique Vivado-probed realizable clocks are forwarded to the full HLS/Vivado implementation sweep.',
    }
    (root/'adaptive_fmax_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if not targets:
        raise SystemExit('No realizable clocks were discovered; inspect the probe report')
    cmd=[sys.executable,'scripts/run_vivado_clock_sweep.py','--config',args.config,'--clocks',*[f'{v:g}' for v in targets],'--out',str(root/'implementation_sweep')]
    if args.continue_on_failure:
        cmd.append('--continue-on-failure')
    print('[ADAPTIVE-FMAX] implementation targets:', ' '.join(f'{v:g}' for v in targets))
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__=='__main__':
    main()
