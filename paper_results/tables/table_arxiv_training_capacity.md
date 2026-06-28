# Training designs. The safe training configuration is explicitly rejected by Vivado due to KV260 LUT capacity.

| Design | Pred LUT | HLS LUT | HLS DSP | Status | Slice LUT util. |
|---|---|---|---|---|---|
| train safe | 84343 | 166965 | 815 | cap reject | 114.18% |
| train fx8 | 105154 | 194087 | 518 | HLS only |  |
