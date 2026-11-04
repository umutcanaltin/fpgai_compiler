"""Independent bounded HLS-loop source model against portable Phi coordinates.

This E1 structural oracle enumerates MatMul/Gemm and Conv source-loop work.
It does not interpret synthesized RTL, source C++ AST, or actual FPGA cycles.
"""
from __future__ import annotations
from collections import Counter
from math import prod
from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturePoint, LoweredOperator


def _bounded_domain(shape: tuple[int, ...], limit: int) -> int:
    if type(limit) is not int or limit <= 0:
        raise ValueError('max_iterations must be a positive integer')
    size = prod(shape)
    if size > limit:
        raise ValueError(f'Logical iteration domain {size} exceeds oracle bound {limit}')
    return size


def _visits_matmul(lo: LoweredOperator):
    m, n, k = lo.iteration_shape
    tm, tn, tk = lo.tile_shape
    pw = lo.replica_count * lo.pe_count
    for mb in range(0, m, tm):
        for nb in range(0, n, tn):
            for ob in range(0, tn, pw * lo.fold_factor):
                for fold in range(lo.fold_factor):
                    for replica in range(lo.replica_count):
                        for pe in range(lo.pe_count):
                            ni = nb + ob + fold * pw + replica * lo.pe_count + pe
                            if ni >= n or ni >= nb + tn:
                                continue
                            for mi in range(mb, min(m, mb + tm)):
                                for kb in range(0, k, tk):
                                    for ki in range(kb, min(k, kb + tk), lo.simd_lanes):
                                        for lane in range(lo.simd_lanes):
                                            kk = ki + lane
                                            if kk < k and kk < kb + tk:
                                                yield (mi, ni, kk), (replica, pe, lane, fold)


def _visits_conv(lo: LoweredOperator):
    bmax, co, ho, wo, ci, kh, kw = lo.iteration_shape
    _, toc, toh, tow, _, _, _ = lo.tile_shape
    pw = lo.replica_count * lo.pe_count
    for b in range(bmax):
        for cb in range(0, co, toc):
            for hb in range(0, ho, toh):
                for wb in range(0, wo, tow):
                    for ob in range(0, toc, pw * lo.fold_factor):
                        for fold in range(lo.fold_factor):
                            for replica in range(lo.replica_count):
                                for pe in range(lo.pe_count):
                                    oc = cb + ob + fold * pw + replica * lo.pe_count + pe
                                    if oc >= co or oc >= cb + toc:
                                        continue
                                    for oh in range(hb, min(ho, hb + toh)):
                                        for ow in range(wb, min(wo, wb + tow)):
                                            for icb in range(0, ci, lo.simd_lanes):
                                                for ky in range(kh):
                                                    for kx in range(kw):
                                                        for lane in range(lo.simd_lanes):
                                                            ic = icb + lane
                                                            if ic < ci:
                                                                yield (b, oc, oh, ow, ic, ky, kx), (replica, pe, lane, fold)


def verify_source_mapping(point: ArchitecturePoint, lo: LoweredOperator, *,
                          max_iterations: int = 100_000) -> dict:
    """Compare independent source-loop traversal to Phi; reject omissions and collisions."""
    if point.operator != lo.name or point.kind != lo.kind or point.iteration_shape != lo.iteration_shape:
        raise ValueError('Source/Phi operator identity mismatch')
    size = _bounded_domain(point.iteration_shape, max_iterations)
    if lo.kind in ('MatMul', 'Gemm'):
        visits = _visits_matmul(lo)
    elif lo.kind == 'Conv':
        visits = _visits_conv(lo)
    else:
        raise ValueError('Only MatMul/Gemm and Conv have independently enumerated HLS schedules')
    visited = set()
    output_axes = Counter()
    for index, assigned in visits:
        if index in visited:
            raise AssertionError(f'HLS work duplicated: {lo.name}, {index}')
        if len(visited) >= size:
            raise AssertionError(f'HLS has more iterations than functional domain: {lo.name}')
        mapped = coordinate(point, index)
        if assigned[:3] != (mapped['replica'], mapped['pe'], mapped['lane']):
            raise AssertionError(f'HLS/Phi spatial assignment mismatch: {lo.name}, {index}')
        if assigned[3] != mapped['reuse_step'][2 if lo.kind in ('MatMul', 'Gemm') else 4]:
            raise AssertionError(f'HLS/Phi temporal folding mismatch: {lo.name}, {index}')
        visited.add(index)
        output_axes[index[1]] += 1
    if len(visited) != size:
        raise AssertionError(f'HLS omitted {size - len(visited)} logical iterations: {lo.name}')
    return {'schema': 'fpgai-research/hls-loop-phi-oracle/v1',
            'operator': lo.name, 'kind': lo.kind, 'logical_iterations': size,
            'source_level_assignments': len(visited),
            'output_coordinates': len(output_axes),
            'source_phi_spatial_and_fold_agree': True,
            'coverage_and_uniqueness': True,
            'evidence': 'E1-independent-source-loop-model-only',
            'limits': 'Independent Python model of source loops, not a C++ AST proof, HLS schedule or post-route evidence'}
