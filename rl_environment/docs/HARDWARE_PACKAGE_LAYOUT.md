# Hardware-RL package layout

Implementation files now live in the existing research subpackages; root
`hardware_*.py` paths are exact module aliases for backward compatibility.

| Boundary | Canonical implementation | Legacy module alias |
| --- | --- | --- |
| Source state and edit contracts | `environments/hardware_space.py` | `hardware_space.py` |
| Native Gym and reward qualification | `environments/hardware_gym.py` | `hardware_gym.py` |
| Shared GNN actor-critic | `agents/hardware_agent.py` | `hardware_agent.py` |
| REINFORCE, PPO, random | `agents/hardware_algorithms.py` | `hardware_algorithms.py` |
| Original Vitis HLS and replay audit | `infrastructure/hardware_native.py` | `hardware_native.py` |
| Tensor-only checkpointing | `infrastructure/hardware_checkpoint.py` | `hardware_checkpoint.py` |
| Source-pinned campaign/journal/audit | `experiments/hardware_campaign.py` | `hardware_campaign.py` |
| Independent pairwise comparison | `evaluation/hardware_compare.py` | `hardware_compare.py` |

All paths above are relative to `rl_environment/`. Existing installed
`fpgai-rl-hardware-*` commands and `python -m rl_environment.hardware_*`
invocations are intentionally preserved via the alias files. Importing a
module never launches vendor tools. In source code, prefer canonical modules,
for example:

```python
from rl_environment.agents.hardware_algorithms import create_hardware_method
from rl_environment.environments.hardware_gym import GymHardwareNativeEnv
from rl_environment.experiments.hardware_campaign import audit_campaign
```

Old imports and monkeypatches resolve to the *same* module object, not a
copied public API. Legacy Gym curriculum code has not been relocated.

**Provenance:** User-provided 2,086-pass/3-skip CPU result and four 2-call
original Vitis HLS audited smoke runs refer to source revision `f8ffe2d`.
They must not be treated as evidence that this new source revision passed.
Re-run CPU tests first, then fresh create-only native smoke with independent
audit. Never overwrite or modify original vendor results.
