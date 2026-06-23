# Scripts

The public FPGAI workflow is exposed through the `fpgai` command-line interface
and the root `main.py` wrapper.

Normal users should prefer commands such as:

    python main.py compile --config configs/examples/inference_compile.yml
    python main.py sweep inspect --config configs/sweeps/precision_selection.yml
    python main.py experiment inspect --config configs/experiments/arxiv_paper.yml

Files in this directory are transitional developer or reproducibility utilities.
They are not the primary public API.

Long-term policy:

- reusable runtime logic should live under `fpgai/runtime/`
- reusable validation logic should live under `fpgai/validation/`
- reusable reporting logic should live under `fpgai/reporting/`
- experiment orchestration should be exposed through `fpgai experiment`
- sweep orchestration should be exposed through `fpgai sweep`
- obsolete one-off scripts should be removed

Do not add new public workflows here. Add package modules and expose them through
the CLI instead.
