"""Two-stage ENMA training on the shared chronological split.

This runner deliberately keeps VAE and flow checkpoints separate.  It uses
the official ENMA regular-grid VAE and masked flow core through
``model.enma`` and only adapts the project's existing GlobalStateDataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.enma import OceanENMAVAE, OceanENMAFlow
from src.data import GlobalStateDataset
from run_metadata import data_contract, public_args


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["vae", "flow"], required=True)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vae-checkpoint", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--validation-every", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--max-validation-batches", type=int)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def grad_norm(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])).item()
    if not torch.isfinite(torch.tensor(norm)):
        raise FloatingPointError("non-finite gradient norm")
    torch.nn.utils.clip_grad_norm_(list(parameters), 1.0)
    return norm


def weighted_mse(pred, target, valid, area):
    weight = valid * area[:, None]
    return ((pred - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def crps_ensemble(samples, target, valid, area):
    # samples [B,S,T,C,H,W], target [B,T,C,H,W]
    weight = valid[:, None] * area[:, None, None]
    first = (samples - target[:, None]).abs().mean(1)
    pair = (samples[:, :, None] - samples[:, None, :]).abs().mean((1, 2)) * 0.5
    return ((first - pair) * weight[:, 0]).sum() / weight[:, 0].sum().clamp_min(1.0)


def val_vae(model, loader, dataset, device, max_batches):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and count >= max_batches: break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            recon, _ = model(batch["x"])
            total += float(weighted_mse(recon, batch["x"], batch["valid"], batch["area"]))
            count += 1
    complete = max_batches is None or count == len(loader)
    return {"complete": complete, "batches": count, "selector": total / max(count, 1) if complete else None}


def val_flow(flow, vae, loader, dataset, device, max_batches, samples=64):
    flow.eval(); vae.eval(); crps = 0.0; count = 0; cover = 0.0; width = 0.0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and count >= max_batches: break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            history = vae.encode(batch["x"], mode=True)
            sample_latent = flow.sample(history, samples=samples)
            b, s, t, h, w, c = sample_latent.shape
            decoded = vae.decode(sample_latent.reshape(b * s, t, h, w, c), batch["target"][:, None].expand(b, s, *batch["target"].shape[1:]).reshape(b * s, t, 5, 90, 180)).reshape(b, s, t, 5, 90, 180)
            target = batch["target"]
            valid = batch["target_valid"]
            crps += float(crps_ensemble(decoded, target, valid, batch["area"]))
            lo, hi = decoded.quantile(0.05, dim=1), decoded.quantile(0.95, dim=1)
            w = valid * batch["area"][:, None]
            cover += float((((target >= lo) & (target <= hi)).float() * w).sum() / w.sum().clamp_min(1.0))
            width += float(((hi - lo) * w).sum() / w.sum().clamp_min(1.0))
            count += 1
    complete = max_batches is None or count == len(loader)
    return {"complete": complete, "batches": count, "selector": crps / max(count, 1) if complete else None, "crps": crps / max(count, 1), "coverage90": cover / max(count, 1), "width90": width / max(count, 1)}


def main():
    a = args()
    if a.output.exists() and not a.resume:
        raise FileExistsError(f"output exists: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); torch.set_num_threads(2)
    device = torch.device(a.device)
    train = GlobalStateDataset(a.release_root, "train", "main", False)
    val = GlobalStateDataset(a.release_root, "validation", "main", False)
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=device.type == "cuda")
    vae = OceanENMAVAE().to(device)
    flow = None
    if a.stage == "vae":
        model = vae
    else:
        if a.vae_checkpoint is None: raise ValueError("--vae-checkpoint is required for flow stage")
        saved_vae = torch.load(a.vae_checkpoint, map_location=device)
        vae.load_state_dict(saved_vae["model"]); vae.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        flow = OceanENMAFlow().to(device); model = flow
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    manifest = {"args": public_args(a), "stage": a.stage, "data_contract": data_contract(train), "model_parameters": sum(p.numel() for p in model.parameters())}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    start, best, best_step = 0, None, None
    if a.resume:
        saved = torch.load(a.output / "checkpoint_last.pt", map_location=device)
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"]); start = int(saved["step"])
        for row in (a.output / "validation.jsonl").read_text().splitlines():
            score = json.loads(row).get("selector")
            if score is not None and (best is None or score < best): best, best_step = score, int(json.loads(row)["step"])
    stream = iter(train_loader); started = time.perf_counter()
    with (a.output / "training.jsonl").open("a" if a.resume else "w") as log, (a.output / "validation.jsonl").open("a" if a.resume else "w") as vlog:
        for step in range(start + 1, a.steps + 1):
            try: batch = next(stream)
            except StopIteration: stream = iter(train_loader); batch = next(stream)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            model.train()
            if a.stage == "vae":
                recon, kl = vae(batch["x"]); loss = weighted_mse(recon, batch["x"], batch["valid"], batch["area"]) + 5e-4 * kl
            else:
                seq = torch.cat([batch["x"], batch["target"]], 1)
                latent = vae.encode(seq, mode=True)
                loss = flow.loss(latent)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
            opt.zero_grad(set_to_none=True); loss.backward(); g = grad_norm(model.parameters()); opt.step()
            row = {"step": step, "loss": float(loss.detach()), "gradient_norm": g, "elapsed_seconds": time.perf_counter() - started}
            if step % a.validation_every == 0 or step == a.steps:
                metric = val_vae(vae, val_loader, val, device, a.max_validation_batches) if a.stage == "vae" else val_flow(flow, vae, val_loader, val, device, a.max_validation_batches)
                row["validation"] = metric; vlog.write(json.dumps({"step": step, **metric}, allow_nan=False) + "\n"); vlog.flush()
                if metric["selector"] is not None and (best is None or metric["selector"] < best):
                    best, best_step = metric["selector"], step
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "validation": metric, "run_config": manifest}, a.output / "checkpoint_best.pt")
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "validation": metric, "run_config": manifest}, a.output / "checkpoint_last.pt")
            log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
            if step % 10 == 0: print(json.dumps(row, allow_nan=False), flush=True)
    report = {"status": "completed", "stage": a.stage, "steps": a.steps, "best_step": best_step, "best_selector": best, "validation_complete": a.max_validation_batches is None, "test_evaluated": False}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
