from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .config import MirageConfig
from .metrics import compare
from .model import DenseBaseline, MirageGenerator


def save_gif(video: torch.Tensor, path: Path, fps: int = 8) -> None:
    frames = ((video[0].clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirage", description="MIRAGE research harness")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "benchmark", "ablate-cache"):
        item = sub.add_parser(name)
        item.add_argument("--config", default="configs/smoke.json")
        item.add_argument("--prompt", default="a silver spacecraft crossing a violet nebula")
        item.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        item.add_argument("--output", default=f"artifacts/{name}")
        item.add_argument("--seed", type=int, default=0)
        if name == "benchmark":
            item.add_argument("--warmup", type=int, default=1)
        if name == "ablate-cache":
            item.add_argument("--threshold", type=float)
    sub.add_parser("doctor")
    return parser


def doctor() -> int:
    report = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        report.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "vram_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
                "bf16": torch.cuda.is_bf16_supported(),
            }
        )
    print(json.dumps(report, indent=2))
    return 0


def run_generate(args) -> int:
    config = MirageConfig.from_json(args.config)
    model = MirageGenerator(config)
    output = model.generate(args.prompt, seed=args.seed, device=args.device)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    save_gif(output.video, target / "sample.gif")
    torch.save({"video": output.video, "audio": output.audio}, target / "sample.pt")
    output.telemetry.save(target / "telemetry.json")
    print(json.dumps(output.telemetry.to_dict(), indent=2))
    return 0


def run_benchmark(args) -> int:
    config = MirageConfig.from_json(args.config)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    candidate_model = MirageGenerator(config)
    torch.manual_seed(args.seed)
    baseline_model = DenseBaseline(config)
    for _ in range(args.warmup):
        candidate_model.generate(args.prompt, seed=args.seed, device=args.device)
        baseline_model.generate(args.prompt, seed=args.seed, device=args.device)
    candidate = candidate_model.generate(args.prompt, seed=args.seed, device=args.device)
    baseline = baseline_model.generate(args.prompt, seed=args.seed, device=args.device)
    result = {
        "mirage": candidate.telemetry.to_dict(),
        "baseline": baseline.telemetry.to_dict(),
        "quality_note": "Models are untrained; cross-architecture perceptual scores are invalid.",
        "speedup": baseline.telemetry.latency_s / candidate.telemetry.latency_s,
        "parameter_compression": baseline.telemetry.parameter_bytes
        / candidate.telemetry.parameter_bytes,
    }
    (target / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_gif(candidate.video, target / "mirage.gif")
    save_gif(baseline.video, target / "baseline.gif")
    print(json.dumps(result, indent=2))
    return 0


def run_cache_ablation(args) -> int:
    config = MirageConfig.from_json(args.config)
    if args.threshold is not None:
        config = MirageConfig(**{**config.to_dict(), "cache_threshold": args.threshold})
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = MirageGenerator(config)
    model.generate(args.prompt, seed=args.seed, device=args.device, use_cache=False)
    reference = model.generate(args.prompt, seed=args.seed, device=args.device, use_cache=False)
    cached = model.generate(args.prompt, seed=args.seed, device=args.device, use_cache=True)
    result = {
        "cache_off": reference.telemetry.to_dict(),
        "cache_on": cached.telemetry.to_dict(),
        "quality_retention": compare(cached.video, reference.video),
        "speedup": reference.telemetry.latency_s / cached.telemetry.latency_s,
        "threshold": config.cache_threshold,
    }
    (target / "cache_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_gif(reference.video, target / "cache_off.gif")
    save_gif(cached.video, target / "cache_on.gif")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "generate":
        return run_generate(args)
    if args.command == "benchmark":
        return run_benchmark(args)
    return run_cache_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
