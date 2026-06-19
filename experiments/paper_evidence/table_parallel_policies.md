# Parallel policy evidence

| design | status | bench_passed | cosine_similarity | policy | pe | simd | unroll | partition | conv_parallel | dense_parallel | weight_partition | parallel_evidence_lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parallel_resource_first | passed | True | 0.98472807 | resource_first | 1 | 1 | 1 | 1 | 1/1 | 1/1 | 1 | 2 |
| parallel_balanced | passed | True | 0.98472807 | balanced | 2 | 2 | 2 | 2 | 2/2 | 2/2 | 4 | 2 |
| parallel_throughput_first | passed | True | 0.98472807 | throughput_first | 2 | 4 | 2 | 4 | 2/4 | 2/4 | 8 | 2 |
| parallel_latency_first_max | passed | True | 0.98472807 | latency_first | 4 | 4 | 4 | 4 | 4/4 | 4/4 | 16 | 2 |
