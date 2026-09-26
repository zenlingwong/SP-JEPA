"""Anonymous, numeric run metadata shared by comparison trainers."""

NUMERIC_ARGUMENTS = (
    "steps", "validation_every", "batch_size", "workers", "seed", "lr",
    "weight_decay", "dim", "depth", "heads", "head_dim", "basis_y",
    "basis_x", "sigreg_weight", "event_weight", "sigreg_temporal_window",
    "internal_height", "internal_width", "deter", "hidden", "stoch",
    "classes", "blocks", "unimix", "free_nats", "dynamics_scale",
    "representation_scale", "validation_sampling_seed", "max_validation_batches",
)


def public_args(args):
    values = {name: getattr(args, name) for name in NUMERIC_ARGUMENTS if hasattr(args, name) and getattr(args, name) is not None}
    if hasattr(args, "method"):
        values["method"] = args.method
    if hasattr(args, "stage"):
        values["stage"] = args.stage
    if hasattr(args, "pretrained_path"):
        values["pretrained"] = args.pretrained_path is not None
    if hasattr(args, "pretrained_checkpoint"):
        values["pretrained"] = args.pretrained_checkpoint is not None
    return values


def data_contract(dataset):
    manifest = dataset.manifest
    return {
        "history": int(manifest["history"]),
        "horizon": int(manifest["horizon"]),
        "splits": {split: [int(bound) for bound in manifest["splits"][split]] for split in ("train", "validation", "test") if split in manifest["splits"]},
        "state_mean": [float(value) for value in manifest["state_mean"]],
        "state_std": [float(value) for value in manifest["state_std"]],
    }
