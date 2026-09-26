"""Evaluate a frozen comparison checkpoint on a declared chronological split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import GlobalStateDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--variant", choices=("pretrained", "scratch"))
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=20260920)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    data = GlobalStateDataset(args.release_root, args.split, "main", False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    if args.method == "enma":
        if args.vae_checkpoint is None or args.flow_checkpoint is None:
            raise ValueError("ENMA requires --vae-checkpoint and --flow-checkpoint")
        from model.enma import OceanENMAVAE, OceanENMAFlow
        from train_enma import crps_ensemble
        from src.metrics import add_field_metrics, empty_metric_totals, finalize_field_metrics

        vae = OceanENMAVAE().to(device)
        flow = OceanENMAFlow().to(device)
        vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True)["model"])
        flow.load_state_dict(torch.load(args.flow_checkpoint, map_location=device, weights_only=True)["model"])
        vae.eval()
        flow.eval()
        model_totals = empty_metric_totals(data.horizon, 5)
        persistence_totals = empty_metric_totals(data.horizon, 5)
        crps_sum, coverage_sum, width_sum, batches = 0.0, 0.0, 0.0, 0
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                history = vae.encode(batch["x"], mode=True)
                latent = flow.sample(history, samples=args.samples, seed=args.sampling_seed)
                b, s, t, h, w, c = latent.shape
                frames = batch["target"][:, None].expand(b, s, *batch["target"].shape[1:]).reshape(b * s, t, 5, 90, 180)
                samples = vae.decode(latent.reshape(b * s, t, h, w, c), frames).reshape(b, s, t, 5, 90, 180)
                mean = samples.mean(1)
                persistence = batch["x"][:, -1:].expand_as(batch["target"])
                std = torch.as_tensor(data.std.reshape(-1), device=device)
                add_field_metrics(model_totals, mean, batch["target"], batch["target_valid"], batch["area"], std, unit_ids=batch["origin_index"])
                add_field_metrics(persistence_totals, persistence, batch["target"], batch["target_valid"], batch["area"], std, unit_ids=batch["origin_index"])
                crps_sum += float(crps_ensemble(samples, batch["target"], batch["target_valid"], batch["area"]))
                lo, hi = samples.quantile(0.05, dim=1), samples.quantile(0.95, dim=1)
                weight = batch["target_valid"] * batch["area"][:, None]
                coverage_sum += float(((((batch["target"] >= lo) & (batch["target"] <= hi)).float() * weight).sum() / weight.sum().clamp_min(1.0)))
                width_sum += float((((hi - lo) * weight).sum() / weight.sum().clamp_min(1.0)))
                batches += 1
        result = {"metrics": finalize_field_metrics(model_totals, persistence_totals), "crps": crps_sum / batches, "coverage90": coverage_sum / batches, "width90": width_sum / batches, "samples": args.samples, "sampling_seed": args.sampling_seed}
        checkpoint_step = {"vae": int(torch.load(args.vae_checkpoint, map_location="cpu", weights_only=True)["step"]), "flow": int(torch.load(args.flow_checkpoint, map_location="cpu", weights_only=True)["step"])}
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.method in {"unet", "convlstm", "fno", "pde_transformer_mse", "neuralom", "dpot", "poseidon", "pde_transformer_pretrained"}:
            from baselines import OceanBaseline
            from train_baseline import validate

            name = args.method
            if name in {"dpot", "poseidon"}:
                name = f"{name}_{args.variant}"
            model = OceanBaseline(name, load_pretrained=False).to(device)
            model.load_state_dict(saved["model"])
            result = validate(model, loader, data, device)
        elif args.method == "climax":
            from model.climax import ClimaX
            from train_climax import validate

            model = ClimaX(**saved["model_init"]).to(device)
            model.load_state_dict(saved["model"])
            result = validate(model, loader, data, device)
        elif args.method == "dreamerv3":
            from model.dreamerv3 import OceanDreamerV3
            from train_dreamerv3 import validate

            model = OceanDreamerV3(**saved["model_init"], **saved["model_kwargs"]).to(device)
            model.load_state_dict(saved["model"])
            result = validate(model, loader, data, device, args.sampling_seed)
        elif args.method in {"lewm", "eawm", "tc_lewm"}:
            from model.ocean import OceanLeWM
            from train_worldmodel import validate

            model = OceanLeWM(**saved["model_init"], write_basis=saved["basis"], weight=saved["weight"], **saved["model_kwargs"]).to(device)
            model.load_state_dict(saved["model"])
            result = validate(model, loader, data, device)
        else:
            raise ValueError(args.method)
        checkpoint_step = int(saved["step"])

    if args.split == "test":
        result.pop("selector", None)
    # This script evaluates the supplied checkpoint only; it never chooses one.
    report = {"method": args.method, "variant": args.variant, "split": args.split, "checkpoint_step": checkpoint_step, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
