# Precision decision matrix

| Design | precision | decision | decision reason | quality metric | quality_metric | max_abs_error | mae | rmse | cosine_similarity | LUT saving vs fx16 % | BRAM saving vs fx16 % | HLS latency cycles | Vivado LUT | Vivado DSP | Vivado BRAM | Power W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I0_baseline_fx16_embedded | fx16_6 | recommended_quality | prediction agreement 1 is near-identical to the reference | prediction_agreement | 1 | 0.459 | 0.0916 | 0.1826 | 0.8394 | 0 | 0 | 329247 | 12708 | 5 | 62.5 | 2.842 |
| I1_precision_fx8_embedded | fx8_3 | not_recommended_for_quality | prediction agreement 0 is below the configured threshold | prediction_agreement | 0 | 0.9062 | 0.1625 | 0.3001 | 0.3162 | 36.9 | 66.4 | 330920 | 8019 | 4 | 21 | 2.786 |
| I2_precision_fx24_embedded | fx24_10 | recommended_quality | prediction agreement 1 is near-identical to the reference | prediction_agreement | 1 | 2.146e-05 | 4.294e-06 | 9.464e-06 | 1 | -32.21 | -34.4 | 329260 | 16801 | 13 | 84 | 2.88 |
