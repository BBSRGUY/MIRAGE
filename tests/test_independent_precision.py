import torch

from mirage.experiments.independent_precision import (
    quantize_fp8_rows,
    quantize_int4_groups,
    quantize_int8_rows,
)


def test_row_quantizers_preserve_small_matrix_closely():
    torch.manual_seed(31)
    weight = torch.randn(16, 32)
    int8, int8_bytes = quantize_int8_rows(weight)
    fp8, fp8_bytes = quantize_fp8_rows(weight)
    assert (int8 - weight).norm() / weight.norm() < 0.02
    assert (fp8 - weight).norm() / weight.norm() < 0.05
    assert int8_bytes == fp8_bytes == weight.numel() + weight.shape[0] * 2


def test_grouped_int4_reports_packed_storage_and_bounded_error():
    torch.manual_seed(37)
    weight = torch.randn(16, 64)
    int4, stored_bytes = quantize_int4_groups(weight)
    assert (int4 - weight).norm() / weight.norm() < 0.15
    assert stored_bytes == weight.numel() // 2 + weight.shape[0] * 2
