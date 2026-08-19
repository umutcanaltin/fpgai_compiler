from fpgai.backends.hls.emit.top_train_cpp import _fpgai_n1_materialize_phase_shared_matmul


CFG = {"architecture": {"network": {"execution": {"mode": "phase_shared"}}}}


BASE = '''
extern "C" void deeplearn() {
  fpgai::matmul_tiled<4, 8, 8, act_t, wgt_t, act_t, acc_t, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2>(BUF0, W_q, BUF1);
  fpgai::matmul_tiled<4, 8, 16, act_t, wgt_t, act_t, acc_t, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2>(BUF0, W_up, BUF2);
  fpgai::matmul_tiled<4, 16, 8, act_t, wgt_t, act_t, acc_t, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2>(BUF3, W_down, BUF4);
  fpgai::matmul_backward_left_accumulate<4, 8, 8, grad_act_t, wgt_t, grad_act_t, acc_t, 2, 1, 1, 1, 2, 2>(DY1, W_q, DX1);
  fpgai::matmul_backward_left_accumulate<4, 16, 8, grad_act_t, wgt_t, grad_act_t, acc_t, 2, 1, 1, 1, 2, 2>(DY2, W_down, DX2);
  fpgai::matmul_weight_grad<4, 8, 8, act_t, grad_act_t, grad_wgt_t, acc_t, 2, 1, 1, 1, 2, 2>(BUF0, DY1, dW_q);
  fpgai::matmul_weight_grad<4, 16, 8, act_t, grad_act_t, grad_wgt_t, acc_t, 2, 1, 1, 1, 2, 2>(BUF3, DY2, dW_down);
}
'''


def test_phase_shared_cross_shape_materializes_bounded_gemm_engines():
    out = _fpgai_n1_materialize_phase_shared_matmul(BASE, raw_cfg=CFG)
    assert "physical=cross_specialization_gemm" in out
    assert "forward_groups=1" in out
    assert "backward_groups=1" in out
    assert "weight_grad_groups=1" in out
    assert "phase_shared_matmul_forward<4, 16, 16" in out
    assert "phase_shared_matmul_forward<4, 16, 16" in out
    assert "(BUF0, W_q, BUF1, 4, 8, 8);" in out
    assert "(BUF0, W_up, BUF2, 4, 8, 16);" in out
    assert "(BUF3, W_down, BUF4, 4, 16, 8);" in out
    assert "phase_shared_matmul_backward_left<4, 16, 8" in out
    assert "phase_shared_matmul_weight_grad<4, 16, 8" in out
    assert "matmul_tiled<4, 8, 8" not in out
    assert "matmul_tiled<4, 8, 16" not in out
    assert "matmul_tiled<4, 16, 8" not in out


def test_phase_shared_does_not_merge_different_hardware_policies():
    src = '''\nextern "C" void deeplearn() {\n  fpgai::matmul_tiled<4, 8, 8, act_t, wgt_t, act_t, acc_t, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2>(A, W_a, B);\n  fpgai::matmul_tiled<4, 8, 16, act_t, wgt_t, act_t, acc_t, 1, 1, 1, 3, 1, 1, 1, 2, 2, 2>(A, W_b, C);\n}\n'''
    out = _fpgai_n1_materialize_phase_shared_matmul(src, raw_cfg=CFG)
    assert "forward_groups=0" in out
    assert "matmul_tiled<4, 8, 8" in out
    assert "matmul_tiled<4, 8, 16" in out


def test_sequential_leaves_matmul_calls_unchanged():
    out = _fpgai_n1_materialize_phase_shared_matmul(
        BASE,
        raw_cfg={"architecture": {"network": {"execution": {"mode": "sequential"}}}},
    )
    assert out == BASE
