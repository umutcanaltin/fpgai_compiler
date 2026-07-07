# P1D training DDR-tiled preload fix

Observed failure:
- `medium_ddr_cifar_training` reached Vitis CSim.
- CSim reported an empty `hls::stream` read immediately after the testbench loaded the DDR runtime preload.
- The testbench printed `mode=ddr_tiled expected_weight_words=11642` and loaded `weights_before_ref.bin`, then called the top in mode 0.

Root cause:
- For `ddr_tiled`/m_axi runtime weights, the testbench already packs preload values into `weights_mem`.
- It still called top mode 0, which is the stream/aux preload mode and expects preload words on `aux_stream`.
- Because m_axi runtime mode intentionally does not push preload values to `aux_stream`, mode 0 read from an empty stream.

Fix:
- Keep mode 0 preload call only for stream-style runtime weight modes.
- For m_axi runtime weights (`ddr`, `dma_ddr`, `external_ddr`, `ddr_tiled`, `ddr_tiled_mutable`), pack `weights_mem` and skip top mode 0.
- Subsequent mode 1/2/3/4/5/6 calls still pass `weights_mem.data()`.
