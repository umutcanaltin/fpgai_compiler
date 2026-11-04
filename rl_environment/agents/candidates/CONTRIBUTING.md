# Adding a candidate DRL agent

1. Open an experiment design entry under `paper/rl_experiments/`, fixing the question, stage, action distributions and equal native-call budget before implementation.
2. Implement an agent separately from the Gym environment and the compiler. Use the `AgentProtocol` in `../api.py`: `choose`, `update`, `save_private`, `load_private`.
3. Respect compiler-derived legal action masks, joint architecture + requested-clock action semantics, and source-attributed numeric/resource/timing qualification. Never train a per-network policy or replace original HLS reward with guessed latency.
4. Add negative CPU tests for invalid action masks, no available actions, stale source import, missing HLS evidence, seed reproducibility, checkpoint integrity, and evaluation without gradients.
5. Register the **tested implementation** in `../registry.py` and verify the native-free CLI lists it. Unimplemented design proposals must remain unregistered.
6. Run source-only tests, then a fresh native smoke with an independent read-only audit before full experiments.
7. Include versioned hyperparameters, algorithm commit, methods description and disclosure of incomplete/negative results in the paper protocol.

A folder alone does not constitute implementation or proof of hardware optimization.
