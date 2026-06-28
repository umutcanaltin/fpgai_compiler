# Training capacity

| design | precision_or_config | prediction_lut | hls_lut | hls_dsp | hls_bram18 | hls_latency_cycles | vivado_status | failure_class | required_slice_luts | available_slice_luts | slice_lut_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| training_kv260_safe_fx16_6 | safe_fx16_6 | 84343 | 166965 | 815 | 77.0 | 114686.0 | vivado_board_capacity_rejected | vivado_impl_failed_board_capacity_lut_overutilized | 133729 | 117120 | 114.18 |
| training_kv260_aggressive_fx8_3 | aggressive_fx8_3 | 105154 | 194087 | 518 | 74.0 | 86082.0 | hls_only |  |  |  |  |
