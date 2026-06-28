# Compact prediction-versus-HLS comparison for representative inference and training designs.

| Design | Mode | Pred LUT | HLS LUT | LUT err. | Pred DSP | HLS DSP | Lat. cyc. |
|---|---|---|---|---|---|---|---|
| KV260 base | infer | 3822 | 5991 | 36.2% | 10 | 10 | 182 |
| fx16 | infer | 4400 | 5056 | 13.0% | 30 | 24 | 90 |
| fx12 | infer | 4114 | 4876 | 15.6% | 24 | 20 | 85 |
| fx8 | infer | 3842 | 4119 | 6.7% | 8 | 8 | 72 |
| x1 | infer | 2802 | 4869 | 42.5% | 6 | 6 | 247 |
| x8 | infer | 7640 | 3584 | 113.2% | 73 | 35 | 80 |
| combined fx8 | infer | 6277 | 3472 | 80.8% | 21 | 7 | 59 |
| train safe | train | 84343 | 166965 | 49.5% | 774 | 815 | 114686 |
| train fx8 | train | 105154 | 194087 | 45.8% | 924 | 518 | 86082 |
