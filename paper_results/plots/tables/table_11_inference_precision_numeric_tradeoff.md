# Inference precision numeric tradeoff

| Design | precision | HLS latency cycles | Vivado LUT | Vivado DSP | Vivado BRAM | Power W | numeric_validation_status | numeric_quality | max abs error | mae | rmse | cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I0_baseline_fx16_embedded | fx16_6 | 329247 | 12708 | 5 | 62.5 | 2.842 | failed_tolerance | failed_numeric_validation | 0.459 | 0.0916 | 0.1826 | 0.8394 |
| I1_precision_fx8_embedded | fx8_3 | 330920 | 8019 | 4 | 21 | 2.786 | failed_tolerance | failed_numeric_validation | 0.9062 | 0.1625 | 0.3001 | 0.3162 |
| I2_precision_fx24_embedded | fx24_10 | 329260 | 16801 | 13 | 84 | 2.88 | passed | passed | 2.146e-05 | 4.294e-06 | 9.464e-06 | 1 |
