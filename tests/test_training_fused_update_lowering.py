from __future__ import annotations

from fpgai.backends.hls.emit.training_fused_update import materialize_dense_fused_update


def _raw_get(raw, path, default=None):
    cur = raw
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _balanced_end(source: str, open_brace: int) -> int:
    depth = 0
    for idx in range(open_brace, len(source)):
        if source[idx] == '{':
            depth += 1
        elif source[idx] == '}':
            depth -= 1
            if depth == 0:
                return idx + 1
    raise AssertionError('unbalanced source')


def test_fused_adam_uses_float_gradient_before_optimizer_state_cast() -> None:
    gradient_call = 'fpgai::dense_backward_params<2, 2>(IN_dense0, OUT_GRAD_dense0, dB_dense0, dW_dense0);'
    source = f'''
static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
{gradient_call}
// FPGAI Adam optimizer update for W_dense0.
for (int i = 0; i < 4; ++i) {{
  float grad_value = (float)dW_dense0[i];
  FPGAI_ADAM_M_W_dense0[i] = (opt_t)(0.9f * (float)FPGAI_ADAM_M_W_dense0[i] + 0.1f * grad_value);
  FPGAI_ADAM_V_W_dense0[i] = (opt_t)(0.999f * (float)FPGAI_ADAM_V_W_dense0[i] + 0.001f * grad_value * grad_value);
  float adam_m_used = (float)FPGAI_ADAM_M_W_dense0[i];
  float adam_v_used = (float)FPGAI_ADAM_V_W_dense0[i];
  float adam_delta = fpgai_adam_delta_shared(adam_m_used, adam_v_used);
  W_dense0[i] = (wgt_t)((float)W_dense0[i] - adam_delta);
}}
OUT_grad_dense0[idx] = (float)dW_dense0[idx];
'''

    spec = {
        'tag': 'dense0',
        'dw': 'dW_dense0',
        'input': 'IN_dense0',
        'output_grad': 'OUT_GRAD_dense0',
        'input_features': 2,
        'output_features': 2,
        'call': gradient_call,
        'call_start': source.index(gradient_call),
        'call_end': source.index(gradient_call) + len(gradient_call),
    }
    raw = {
        'training': {
            'optimizer': {'type': 'adam'},
            'batch': {'mode': 'direct', 'size': 1},
            'gradient_accumulation': {'steps': 1},
            'gradients': {'export_policy': 'recompute'},
        }
    }

    lowered = materialize_dense_fused_update(
        source,
        raw_cfg=raw,
        raw_get=_raw_get,
        parameter_gradient_policy=lambda _: ('fused_update', 'recompute'),
        dense_gradient_specs=lambda _: [spec],
        balanced_brace_end=_balanced_end,
    )

    assert 'const float fused_grad_value = ((float)IN_dense0[input_index] * (float)OUT_GRAD_dense0[output_index]);' in lowered
    assert 'float grad_value = fused_grad_value;' in lowered
    assert 'FPGAI_ADAM_M_W_dense0[gradient_index]' in lowered
    assert 'W_dense0[gradient_index]' in lowered
    assert 'static grad_wgt_t dW_dense0[4]' not in lowered
    assert gradient_call not in lowered
    assert 'OUT_grad_dense0[idx] = (float)(grad_wgt_t)' in lowered
    assert 'adam_gradient_arithmetic=float_before_state_cast' in lowered


def test_fused_adam_selected_probe_is_csim_only() -> None:
    from fpgai.backends.hls.emit.training_trace_probes import instrument_fused_dense_adam_probe

    source = '''
#include <cmath>
// FPGAI fused_update Adam update for W_dense0; no complete or tiled dW_dense0 buffer is materialized.
for (int gradient_index = 0; gradient_index < 4; ++gradient_index) {
#pragma HLS PIPELINE II=1
  const int output_index = gradient_index / 2;
  const int input_index = gradient_index % 2;
  const float fused_grad_value = ((float)IN_dense0[input_index] * (float)OUT_GRAD_dense0[output_index]);
  float grad_value = fused_grad_value;
  FPGAI_ADAM_M_W_dense0[gradient_index] = (opt_t)(0.9f * (float)FPGAI_ADAM_M_W_dense0[gradient_index] + 0.1f * grad_value);
  FPGAI_ADAM_V_W_dense0[gradient_index] = (opt_t)(0.999f * (float)FPGAI_ADAM_V_W_dense0[gradient_index] + 0.001f * grad_value * grad_value);
  float adam_m_used = (float)FPGAI_ADAM_M_W_dense0[gradient_index];
  float adam_v_used = (float)FPGAI_ADAM_V_W_dense0[gradient_index];
  float adam_delta = fpgai_adam_delta_shared(adam_m_used, adam_v_used);
  W_dense0[gradient_index] = (wgt_t)((float)W_dense0[gradient_index] - adam_delta);
}
'''
    raw = {"validation": {"numeric": {"probes": {
        "enabled": True,
        "selectors": [{"operator": "dense0", "parameter": "weight", "tensor_index": [1, 1]}],
    }}}}
    instrumented = instrument_fused_dense_adam_probe(source, raw_cfg=raw)
    assert 'extern "C" float fpgai_training_probe_values[16];' in instrumented
    assert 'if (gradient_index == 3)' in instrumented
    assert 'fpgai_training_probe_values[0] = (float)IN_dense0[input_index];' in instrumented
    assert 'fpgai_training_probe_values[1] = (float)OUT_GRAD_dense0[output_index];' in instrumented
    assert 'fpgai_training_probe_values[8] = (float)W_dense0[gradient_index];' in instrumented
    assert 'fpgai_training_probe_values[9] = 1.0f;' in instrumented
    assert 'fpgai_training_probe_values[10] = 1.0f;' in instrumented
    assert 'fpgai_training_probe_values[11] = 1.0f;' in instrumented
    assert instrumented.count('#ifndef __SYNTHESIS__') >= 5


def test_fused_adam_lowers_direct_update_when_mode4_update_precedes_it() -> None:
    gradient_call = 'fpgai::dense_backward_params<2, 2>(IN_dense0, OUT_GRAD_dense0, dB_dense0, dW_dense0);'
    update = '''// FPGAI Adam optimizer update for W_dense0.
for (int i = 0; i < 4; ++i) {
  float grad_value = (float)dW_dense0[i];
  FPGAI_ADAM_M_W_dense0[i] = (opt_t)(0.9f * (float)FPGAI_ADAM_M_W_dense0[i] + 0.1f * grad_value);
  FPGAI_ADAM_V_W_dense0[i] = (opt_t)(0.999f * (float)FPGAI_ADAM_V_W_dense0[i] + 0.001f * grad_value * grad_value);
  float adam_m_used = (float)FPGAI_ADAM_M_W_dense0[i];
  float adam_v_used = (float)FPGAI_ADAM_V_W_dense0[i];
  float adam_delta = fpgai_adam_delta_shared(adam_m_used, adam_v_used);
  W_dense0[i] = (wgt_t)((float)W_dense0[i] - adam_delta);
}'''
    source = f'''static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
{gradient_call}
if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS || mode == 4) {{
  {update}
  return;
}}
{update}
OUT_grad_dense0[idx] = (float)dW_dense0[idx];
'''
    spec = {
        'tag': 'dense0', 'dw': 'dW_dense0', 'input': 'IN_dense0',
        'output_grad': 'OUT_GRAD_dense0', 'input_features': 2, 'output_features': 2,
        'call': gradient_call, 'call_start': source.index(gradient_call),
        'call_end': source.index(gradient_call) + len(gradient_call),
    }
    raw = {'training': {'optimizer': {'type': 'adam'}, 'batch': {'mode': 'direct', 'size': 1},
                        'gradient_accumulation': {'steps': 1}, 'gradients': {'export_policy': 'recompute'}}}
    lowered = materialize_dense_fused_update(
        source, raw_cfg=raw, raw_get=_raw_get,
        parameter_gradient_policy=lambda _: ('fused_update', 'recompute'),
        dense_gradient_specs=lambda _: [spec], balanced_brace_end=_balanced_end,
    )
    mode4_end = lowered.index('return;')
    fused_pos = lowered.index('// FPGAI fused_update Adam update for W_dense0;')
    assert fused_pos > mode4_end
    assert lowered[:mode4_end].count('// FPGAI Adam optimizer update for W_dense0.') == 1
