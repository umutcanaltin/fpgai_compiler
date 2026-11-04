#!/usr/bin/env bash
# R2z/R3a mixed CNN→Dense original SOURCE publication gate.
# Runs Python/GCC only; never Vitis or Vivado; never overwrites old studies.
set -euo pipefail

if [[ ! -f pyproject.toml || ! -d src/fpgai_research ]]; then
  echo "[FPGAI] Run from the fpgai_research repository root." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[FPGAI] Refusing original source experiment: tracked Git files are dirty." >&2
  exit 2
fi
STUDY="$(printenv FPGAI_R2Z_STUDY || true)"
if [[ -z "$STUDY" ]]; then
  STUDY="build/r2z_conv_dense_original_source_6_v1"
fi
if [[ "$STUDY" == "/" || "$STUDY" == ".." ]]; then
  echo "[FPGAI] Unsafe R2z original output directory." >&2
  exit 2
fi

printf '\n[FPGAI] Pinned source revision: '
git rev-parse HEAD
printf '\n[FPGAI] Step 1 — compile and focused regression.\n'
python -m compileall -q src/fpgai_research experiments/compiler_ir tests
python -m pytest -q \
  tests/test_cst_r2z_conv_dense_neural_network.py \
  tests/test_cst_r2z_neural_campaign.py \
  tests/test_cst_r3a_conv_dense_native_hls.py \
  tests/test_cst_r3b_publication_handoff.py \
  tests/test_cst_r3c_cnn_publication_tables.py \
  tests/test_cst_r3d_c1c2_finite_contract_ablation.py \
  tests/test_cst_r3e_c1_fanout_counterexamples.py \
  tests/test_cst_r3f_c2_variable_graph_coverage.py \
  tests/test_cst_r3g_c1_independent_event_semantics.py \
  tests/test_cst_compiled_action_catalog.py \
  tests/test_cst_r3h_c2_compiled_once_selection.py \
  tests/test_cst_r3i_external_allo_capability.py \
  tests/test_cst_r3j_c1_heldout_semantics.py \
  tests/test_cst_r3k_c2_target_limit_intervention.py \
  tests/test_cst_r3l_c1c2_publication_release.py \
  tests/test_cst_r2y_controlled_physical_pilot.py

printf '\n[FPGAI] Step 2 — complete Python regression.\n'
python -m pytest -q

if [[ -e "$STUDY" || -L "$STUDY" ]]; then
  printf '\n[FPGAI] Existing original R2z study preserved: %s\n' "$STUDY"
else
  printf '\n[FPGAI] Step 3 — 6 source packages × 2 actual GCC probes (no Vitis).\n'
  python -m fpgai_research.experiments.cst_r2z_conv_dense_network_campaign \
    --output "$STUDY"
fi

printf '\n[FPGAI] Step 4 — read-only source SHA/denominator audit.\n'
python -m fpgai_research.experiments.cst_r2z_conv_dense_network_campaign \
  --study "$STUDY"

printf '\n[FPGAI] Step 5 — read-only HLS qualification and paired UNKNOWN reporting.\n'
python -m fpgai_research.experiments.cst_r3a_conv_dense_native_hls \
  --audit --study "$STUDY"
python -m fpgai_research.experiments.cst_r3a_conv_dense_native_hls \
  --contrasts --study "$STUDY"

printf '\n[FPGAI] Step 6 — DRY RUN, zero vendor calls even if case_000 is already attempted.\n'
python -m fpgai_research.experiments.cst_r3a_conv_dense_native_hls \
  --execute --study "$STUDY" --start-case 0 --max-cases 1

printf '\n[FPGAI] Step 7 — read-only R3b publication evidence snapshot.\n'
python -m fpgai_research.experiments.cst_r3b_publication_handoff \
  --study "$STUDY"
printf '\n[FPGAI] Source gate finished. Native E2/E3 and physical E4 remain separate.\n'
