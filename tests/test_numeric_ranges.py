"""Signed precision preflight has E0 semantics and catches range mistakes."""
import numpy as np
import pytest
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel
from fpgai_research.validation.ranges import FixedFormat, audit_ranges
from fpgai_research.validation.reference import evaluate


def model():
    return FunctionalModel(
        (Tensor('x', (1, 2)), Tensor('w', (2, 1), data=(2., 3.)),
         Tensor('z', (1, 1)), Tensor('y', (1, 1))),
        (Operator('mm', 'MatMul', ('x', 'w'), ('z',)),
         Operator('id', 'Identity', ('z',), ('y',))),
        ('x',), ('y',))


def test_ranges_saturate_only_at_required_widths():
    samples = np.array([[3., 2.], [-2., -3.]], dtype=np.float32)
    results = audit_ranges(model(), samples, (FixedFormat(8, 4), FixedFormat(16, 6)))
    assert results['samples'] == 2
    assert results['not_bit_exact'] is True
    assert results['formats']['W8_I4']['possible_saturation'] is True
    assert results['formats']['W16_I6']['possible_saturation'] is False
    assert results['tensor_statistics']['z']['max'] == 12.
    assert results['tensor_statistics']['w']['role'] == 'constant'
    assert evaluate(model(), {'x': samples[:1]})['y'].shape == (1, 1)


def test_reject_duplicate_formats_and_float64_inputs():
    samples = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match='unique'):
        audit_ranges(model(), samples, (FixedFormat(8, 4), FixedFormat(8, 4)))
    with pytest.raises(ValueError, match='float32'):
        audit_ranges(model(), samples.astype(np.float64), (FixedFormat(8, 4),))


def test_range_boundaries_and_budget():
    fmt=FixedFormat(8, 4)
    assert fmt.minimum == -8
    assert fmt.maximum == 7.9375
    with pytest.raises(ValueError, match='width'):
        FixedFormat(7, 8)
    with pytest.raises(ValueError, match='budget'):
        audit_ranges(model(), np.ones((2, 2), np.float32), (fmt,), max_total_elements=2)
