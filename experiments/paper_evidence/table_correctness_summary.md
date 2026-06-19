# Correctness summary

| experiment | design | status | bench_passed | cosine_similarity | max_abs_error | mean_abs_error | rmse | argmax_match |
|---|---|---|---|---|---|---|---|---|
| sprint12a_ddr_memory_strategy | memory_on_chip_balanced | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12a_ddr_memory_strategy | memory_streaming_balanced | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12a_ddr_memory_strategy | memory_external_ddr_balanced | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12a_ddr_memory_strategy | memory_external_ddr_latency_first | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12b_memory_binding_strategy | memory_on_chip_bram_baseline | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12b_memory_binding_strategy | memory_bram_saver_balanced | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12b_memory_binding_strategy | memory_uram_first_balanced | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12b_memory_binding_strategy | memory_uram_first_latency_first | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12c_parallel_policy_strategy | parallel_resource_first | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12c_parallel_policy_strategy | parallel_balanced | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12c_parallel_policy_strategy | parallel_throughput_first | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
| sprint12c_parallel_policy_strategy | parallel_latency_first_max | passed | True | 0.98472807 | 0.04679399 | 0.01361360 | 0.01913300 | True |
