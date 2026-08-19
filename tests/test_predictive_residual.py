import torch

from mirage.temporal.predictor import fit_predictor


def test_predictor_converges_on_controlled_residuals():
    torch.manual_seed(8)
    transform = torch.randn(8, 8) * 0.1

    def sample(scale):
        delta = torch.randn(2, 6, 8) * scale
        previous = torch.randn(2, 6, 8)
        target = previous + delta @ transform
        return delta, previous, target, scale

    train = [sample(value) for value in (0.2, 0.4, 0.6, 0.8)]
    validation = [sample(value) for value in (0.3, 0.7)]
    fitted = fit_predictor(
        8, train, validation, steps=500, learning_rate=0.01, seed=3, bottleneck=8
    )
    assert fitted.validation_relative_error < 0.08
    assert fitted.validation_cosine > 0.995
