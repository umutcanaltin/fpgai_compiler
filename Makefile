# Local, explicitly invoked source-only workflows.
# None of these targets invokes Vitis, Vivado, downloads models or publishes code.
PYTHON ?= python

.PHONY: help install-dev example example-cnn example-strided example-two-conv example-mlp test test-examples test-compiler test-gym release-check release-links release-preview-plan fpga-env-check fpga-prepare-hls fpga-inspect research-inventory research-audit-tests research-c2-review research-c1-e1-review research-hls-vivado-review research-r4w-audit research-r4x-feasible-latency research-r4y-plan research-r4y-prepare research-r4y-audit research-r4z-native-prepare research-r4z-native-dry-run research-r4z-native-audit

help:
	@echo "install-dev          Install editable compiler with ONNX and test extras"
	@echo "example              Generate a tiny synthetic MatMul-ReLU graph"
	@echo "example-cnn          Generate/compile Conv-Pool-Dense network"
	@echo "example-strided      Generate/compile stride-2 Conv-Dense network"
	@echo "example-two-conv     Generate/compile two-Conv Softmax network"
	@echo "example-mlp          Generate/compile a two-layer dense network"
	@echo "test                 Run full local regression (no vendor tools)"
	@echo "test-examples        Verify public ONNX and CNN examples"
	@echo "test-compiler        Run focused source compiler regressions"
	@echo "test-gym             Run separate frozen Gym tests (no vendor)"
	@echo "research-inventory  Read-only tracked-file SHA and Python import/owner map"
	@echo "research-audit-tests Run source inventory, hygiene and C2 read-only tests"
	@echo "research-c2-review  Read-only original source-compiler cost comparison (no new calls)"
	@echo "research-c1-e1-review Read-only SHA-pinned original full 54-profile GCC E1 source-slot results"
	@echo "research-hls-vivado-review Original Vitis HLS vs routed Vivado matched source Pareto (read-only)"
	@echo "research-r4w-audit   New isolated 16/20/24ns routed-clock qualification (read-only)"
	@echo "research-r4x-feasible-latency Joint timing+DRC-valid-only HLS-derived latency ns (read-only)"
	@echo "research-r4y-plan     Prospective isolated four-arm 24ns source plan (no tools)"
	@echo "research-r4y-prepare  NEW create-only 24ns Vitis SOURCE packages (no vendor)"
	@echo "research-r4y-audit    Read-only SHA verification of R4y packages"
	@echo "research-r4z-native-prepare  NEW four-case Vitis source copy (no vendor)"
	@echo "research-r4z-native-dry-run Dry-run four Vitis attempts (no vendor)"
	@echo "research-r4z-native-audit   Read-only source-pinned HLS receipt audit"
	@echo "release-links        Check all public preview documentation links"
	@echo "release-preview-plan Show nonpublishable public source allowlist"
	@echo "release-check        Read-only owner/IP/license/history release preflight"
	@echo "fpga-env-check       Read-only HLS/Vivado/XRT environment discovery"
	@echo "fpga-prepare-hls     Prepare new synthetic CNN HLS source/testbench (NO vendor execution)"
	@echo "fpga-inspect         Read-only inventory of HLS vs board deployment evidence"

install-dev:
	$(PYTHON) -m pip install -e '.[onnx,test]'

example:
	mkdir -p build
	$(PYTHON) examples/make_toy_onnx.py build/toy.onnx
	$(PYTHON) -m fpgai_research.cli build/toy.onnx -o build/example-toy

example-cnn:
	mkdir -p build/gallery
	$(PYTHON) examples/make_neural_network_onnx.py cnn_pool_dense build/gallery/cnn_pool_dense.onnx
	$(PYTHON) examples/compile_neural_network.py build/gallery/cnn_pool_dense.onnx --output build/gallery/cnn_pool_dense_hls

example-strided:
	mkdir -p build/gallery
	$(PYTHON) examples/make_neural_network_onnx.py cnn_strided_dense build/gallery/cnn_strided_dense.onnx
	$(PYTHON) examples/compile_neural_network.py build/gallery/cnn_strided_dense.onnx --output build/gallery/cnn_strided_dense_hls

example-two-conv:
	mkdir -p build/gallery
	$(PYTHON) examples/make_neural_network_onnx.py cnn_two_conv_softmax build/gallery/cnn_two_conv_softmax.onnx
	$(PYTHON) examples/compile_neural_network.py build/gallery/cnn_two_conv_softmax.onnx --output build/gallery/cnn_two_conv_softmax_hls

example-mlp:
	mkdir -p build/gallery
	$(PYTHON) examples/make_mlp_onnx.py build/gallery/tiny_mlp.onnx
	$(PYTHON) -m fpgai_research.cli build/gallery/tiny_mlp.onnx -o build/gallery/tiny_mlp_hls

# This prepare step does not execute Vitis or Vivado; the generated Tcl
# must be inspected before any licensed native invocation.
fpga-env-check:
	$(PYTHON) scripts/check_amd_environment.py

fpga-prepare-hls:
	mkdir -p build/native_demo
	$(PYTHON) examples/make_neural_network_onnx.py cnn_pool_dense build/native_demo/model.onnx
	$(PYTHON) examples/make_sample_input.py build/native_demo/model.onnx build/native_demo/input.npy
	$(PYTHON) -m fpgai_research.toolchains.workflow prepare \
	  --onnx build/native_demo/model.onnx \
	  --input build/native_demo/input.npy \
	  --case build/native_demo/hls_case \
	  --part xck26-sfvc784-2LV-c --clock-ns 10 --tolerance 0.01

fpga-inspect:
	$(PYTHON) scripts/check_fpga_deployment.py --case build/native_demo/hls_case --target kv260

test:
	$(PYTHON) -m pytest -q

test-examples:
	$(PYTHON) -m pytest -q tests/test_public_neural_network_examples.py

test-compiler:
	$(PYTHON) -m pytest -q tests/test_cst_pipeline.py tests/test_cst_row_chain.py tests/test_cst_r3j_c1_heldout_semantics.py tests/test_cst_r3k_c2_target_limit_intervention.py

test-gym:
	$(PYTHON) -m pytest -q tests/test_online_post_hls_env.py tests/test_online_post_hls_audit.py tests/test_online_post_hls_report.py

release-links:
	$(PYTHON) -m scripts.check_public_docs --profile compiler

release-preview-plan:
	$(PYTHON) scripts/prepare_public_preview.py --profile compiler --plan

release-check:
	$(PYTHON) scripts/check_open_source_readiness.py

# Read-only, contributor-friendly research and publication checks.
# These targets never create native results or invoke Vitis/Vivado.
research-inventory:
	$(PYTHON) scripts/audit_source_inventory.py

research-audit-tests:
	$(PYTHON) -m pytest -q tests/test_audit_source_inventory.py tests/test_repository_publication_hygiene.py tests/test_cst_paper_c2_original_readonly_analysis.py tests/test_cst_paper_c1_original_e1_readonly_descriptives.py tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py tests/test_cst_paper_r4w_feasible_clock_sweep.py tests/test_cst_paper_r4x_feasible_only_core_latency.py tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py tests/test_cst_paper_r4z_24ns_four_arm_native.py tests/test_cst_graph_phi_independent_issue.py tests/test_cst_graph_phi_source.py tests/test_cst_nchw_graph_contract.py tests/test_public_docs_links.py tests/test_public_release_preview.py

research-c2-review:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_c2_original_readonly_analysis --study build/paper_c2_fair_source_cost_20261106_01

# Source-hash-pinned, NO new compilation or vendor calls. Never re-run the original GCC output.
research-c1-e1-review:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_c1_original_e1_readonly_descriptives --study build/paper_c1_e1_cpp_bridge_full_20261106_01 --summary-only


# All three targets READ existing author-original studies. They do not run GCC,
# Vitis or Vivado, delete attempt locks, modify reports or re-run original studies.
R4_SOURCE_ROOT = build/r4k_source_schedule_matched_storage_20261106_01
R4_OOC_INPUTS = build/r4o_matched_vivado_ooc_prepare_20261106_01
R4_ORIGINAL_NATIVE = build/r4p_matched_vivado_ooc_native_20261106_01
R4_ORIGINAL_R4D = build/r4d_numeric_format_source_20261106_01
R4_ORIGINAL_R4A = build/r4a_row_retained_paper_20261106_01
R4_NEW_FEASIBILITY = build/r4w_valid_clock_sweep_20261106_01

research-hls-vivado-review:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4v_vitis_vivado_joint_compare \
	  --r4p-study $(R4_ORIGINAL_NATIVE) \
	  --r4o-study $(R4_OOC_INPUTS) \
	  --r4k-study $(R4_SOURCE_ROOT) \
	  --original-r4d $(R4_ORIGINAL_R4D) \
	  --original-r4a $(R4_ORIGINAL_R4A)

research-r4w-audit:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4w_feasible_clock_sweep \
	  --study $(R4_NEW_FEASIBILITY) --audit-native \
	  --r4o-study $(R4_OOC_INPUTS) \
	  --r4k-study $(R4_SOURCE_ROOT) \
	  --original-r4d $(R4_ORIGINAL_R4D) \
	  --original-r4a $(R4_ORIGINAL_R4A)

research-r4x-feasible-latency:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4x_feasible_only_core_latency \
	  --study $(R4_NEW_FEASIBILITY) \
	  --r4o-study $(R4_OOC_INPUTS) \
	  --r4k-study $(R4_SOURCE_ROOT) \
	  --original-r4d $(R4_ORIGINAL_R4D) \
	  --original-r4a $(R4_ORIGINAL_R4A)

# R4y source-only, distinct from old one-time R4k/R4l and R4w artifacts.
# All commands are deliberately foreground. The new folder is create-only.
R4Y_NEW_SOURCE = build/r4y_24ns_four_arm_sources_20261106_01

.PHONY: research-r4y-plan research-r4y-prepare research-r4y-audit

research-r4y-plan:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4y_24ns_pipeline_source_prepare --plan

research-r4y-prepare:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4y_24ns_pipeline_source_prepare \
	  --prepare-output $(R4Y_NEW_SOURCE) \
	  --r4k-study $(R4_SOURCE_ROOT) \
	  --original-r4d $(R4_ORIGINAL_R4D) \
	  --original-r4a $(R4_ORIGINAL_R4A)

research-r4y-audit:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4y_24ns_pipeline_source_prepare \
	  --audit-study $(R4Y_NEW_SOURCE) \
	  --r4k-study $(R4_SOURCE_ROOT) \
	  --original-r4d $(R4_ORIGINAL_R4D) \
	  --original-r4a $(R4_ORIGINAL_R4A)

# R4z copies validated R4y source inputs into a NEW ignored study root.
# These targets never execute Vitis/GCC/Vivado. The explicit native CLI has
# separate --execute-native AND --confirm-isolated-native gates.
R4Z_NEW_NATIVE = build/r4z_24ns_four_arm_native_20261107_01

research-r4z-native-prepare:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4z_24ns_four_arm_native \
	  --prepare-output $(R4Z_NEW_NATIVE) \
	  --source-study $(R4Y_NEW_SOURCE) \
	  --r4k-study $(R4_SOURCE_ROOT) \
	  --original-r4d $(R4_ORIGINAL_R4D) \
	  --original-r4a $(R4_ORIGINAL_R4A)

research-r4z-native-dry-run:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4z_24ns_four_arm_native \
	  --study $(R4Z_NEW_NATIVE)

research-r4z-native-audit:
	$(PYTHON) -m fpgai_research.experiments.cst_paper_r4z_24ns_four_arm_native \
	  --study $(R4Z_NEW_NATIVE) --audit-native

