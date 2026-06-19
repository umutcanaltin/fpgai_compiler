# Memory mode evidence

| design | status | bench_passed | cosine_similarity | weight_mode | has_stream_port | has_ddr_port | has_m_axi_weights |
|---|---|---|---|---|---|---|---|
| memory_on_chip_balanced | passed | True | 0.98472807 | embedded | False | False | False |
| memory_streaming_balanced | passed | True | 0.98472807 | stream | True | True | False |
| memory_external_ddr_balanced | passed | True | 0.98472807 | ddr | True | True | True |
| memory_external_ddr_latency_first | passed | True | 0.98472807 | ddr | True | True | True |
