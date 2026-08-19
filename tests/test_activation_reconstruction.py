import torch

from mirage.compression.activation_fit import activation_metrics


def test_exact_activation_reconstruction():
    torch.manual_seed(2)
    inputs, weight = torch.randn(2, 7, 4), torch.randn(5, 4)
    metrics = activation_metrics(inputs, weight, weight.clone())
    assert metrics.relative_activation_error == 0
    assert metrics.cosine_similarity > 0.999999
    assert metrics.token_count == 14
