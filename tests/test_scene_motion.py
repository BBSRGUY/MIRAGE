import torch

from mirage.temporal.scene_motion_analysis import analyze_scene_motion


def test_scene_motion_decomposition():
    torch.manual_seed(5)
    scene = torch.randn(1, 1, 6, 8)
    motion = torch.randn(1, 4, 6, 8) * 0.05
    report = analyze_scene_motion(scene + motion, [1, 2, 8], 0.8)
    assert report["temporal_mean"]["reconstruction_error"] < 1e-7
    assert report["temporal_pca"]["1"]["explained_energy"] > 0.9
    assert report["temporal_pca"]["8"]["status"] == "skipped_invalid"
