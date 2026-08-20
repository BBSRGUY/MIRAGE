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
    for name in (
        "teacher-extract",
        "m2-basis-sweep",
        "m2-activation-sweep",
        "m2-temporal-probe",
        "m2-cache-analysis",
        "m2-predictor-fit",
        "m2-scene-motion",
        "m2-report",
        "m21-adaptive-fit",
        "m21-adaptive-activation-sweep",
        "m21-delta-spectrum",
        "m21-report",
        "m22-residual-spectrum",
        "m22-sparse-study",
        "m22-independent-study",
        "m22-allocate",
        "m22-block-replay",
        "m22-report",
    ):
        item = sub.add_parser(name)
        item.add_argument("--config", default="configs/m2_ltx_teacher.json")
        if name in {
            "m2-basis-sweep",
            "m2-activation-sweep",
            "m2-predictor-fit",
            "m21-adaptive-fit",
            "m21-adaptive-activation-sweep",
            "m22-residual-spectrum",
            "m22-sparse-study",
            "m22-independent-study",
            "m22-block-replay",
        }:
            item.add_argument("--device")
        if name == "m21-adaptive-activation-sweep":
            item.add_argument("--behavior-fit", action="store_true")
    activation = sub.add_parser("m2-activation-fit")
    activation.add_argument("--config", default="configs/m2_ltx_teacher.json")
    activation.add_argument("--artifact", required=True)
    activation.add_argument("--device")
    activation.add_argument("--behavior-fit", action="store_true")
    m3_train = sub.add_parser("m3-train")
    m3_train.add_argument("--config", default="configs/m3_mirage_s.json")
    m3_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    m3_eval = sub.add_parser("m3-eval")
    m3_eval.add_argument("--config", default="configs/m3_mirage_s.json")
    m3_eval.add_argument("--checkpoint", required=True)
    m3_eval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    m3_features = sub.add_parser("m3-build-teacher-features")
    m3_features.add_argument("--config", default="configs/m3_mirage_s.json")
    m3_features.add_argument("--m2-root", default="artifacts/m2/ltx25_22b")
    m3_report = sub.add_parser("m3-report")
    m3_report.add_argument("--config", default="configs/m3_mirage_s.json")
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


def run_m2(args: argparse.Namespace) -> int:
    from .m2_config import M2Config

    config = M2Config.from_json(args.config)
    config.validate()
    if args.command == "teacher-extract":
        from .experiments.teacher_extract import run_teacher_extraction

        result = run_teacher_extraction(config)
    elif args.command == "m2-basis-sweep":
        from .experiments.basis_sweep import run_basis_sweep

        result = run_basis_sweep(config, args.device)
    elif args.command == "m2-activation-fit":
        from .experiments.activation_reconstruction import run_activation_reconstruction

        result = run_activation_reconstruction(
            config, args.artifact, behavior_fit=args.behavior_fit, device=args.device
        )
    elif args.command == "m2-activation-sweep":
        from .experiments.activation_reconstruction import run_activation_sweep

        result = run_activation_sweep(config, args.device)
    elif args.command == "m2-temporal-probe":
        from .experiments.temporal_redundancy import run_temporal_probe

        result = run_temporal_probe(config)
    elif args.command == "m2-cache-analysis":
        from .experiments.cache_or_predict import run_cache_analysis

        result = run_cache_analysis(config)
    elif args.command == "m2-predictor-fit":
        from .experiments.cache_or_predict import run_predictor_fit

        result = run_predictor_fit(config, args.device)
    elif args.command == "m2-scene-motion":
        from .experiments.scene_motion_probe import run_scene_motion_probe

        result = run_scene_motion_probe(config)
    elif args.command == "m21-adaptive-fit":
        from .experiments.adaptive_compression import run_adaptive_compression

        result = run_adaptive_compression(config, args.device)
    elif args.command == "m21-adaptive-activation-sweep":
        from .experiments.adaptive_compression import run_adaptive_activation_sweep

        result = run_adaptive_activation_sweep(
            config, args.device, behavior_fit=args.behavior_fit
        )
    elif args.command == "m21-delta-spectrum":
        from .experiments.delta_spectrum import run_delta_spectrum

        result = run_delta_spectrum(config)
    elif args.command == "m21-report":
        from .experiments.recovery_report import generate_m21_report

        result = generate_m21_report(config)
    elif args.command == "m22-residual-spectrum":
        from .experiments.residual_spectrum import run_residual_spectrum

        result = run_residual_spectrum(config, args.device)
    elif args.command == "m22-sparse-study":
        from .experiments.sparse_residual import run_sparse_residual_study

        result = run_sparse_residual_study(config, args.device)
    elif args.command == "m22-independent-study":
        from .experiments.independent_precision import run_independent_precision_study

        result = run_independent_precision_study(config, args.device)
    elif args.command == "m22-allocate":
        from .experiments.heterogeneous_allocator import run_heterogeneous_allocation

        result = run_heterogeneous_allocation(config)
    elif args.command == "m22-block-replay":
        from .experiments.block_replay import run_block_replay

        result = run_block_replay(config, args.device)
    elif args.command == "m22-report":
        from .experiments.m22_decision import generate_m22_decision

        result = generate_m22_decision(config)
    else:
        from .experiments.report import generate_m2_report

        result = generate_m2_report(config)
    print(json.dumps(result, indent=2, default=str))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command.startswith("m3-"):
        from .m3_config import M3Config

        config = M3Config.from_json(args.config)
        if args.command == "m3-train":
            from .training import train_m3

            result = train_m3(config, args.device)
        elif args.command == "m3-eval":
            from .training import evaluate_m3

            result = evaluate_m3(config, args.checkpoint, args.device)
        elif args.command == "m3-build-teacher-features":
            from .experiments.m3_teacher_features import build_m3_teacher_features

            result = build_m3_teacher_features(config, args.m2_root)
        else:
            from .experiments.m3_status import generate_m3_status

            result = generate_m3_status(config)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "doctor":
        return doctor()
    if args.command == "generate":
        return run_generate(args)
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "ablate-cache":
        return run_cache_ablation(args)
    return run_m2(args)


if __name__ == "__main__":
    raise SystemExit(main())
