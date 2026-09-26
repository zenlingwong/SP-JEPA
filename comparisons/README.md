# Comparison methods

This directory provides 14 methods in 17 training configurations for the three-month global forecast task. Each method has `train.py` and `evaluate.py` entries; shared readers, trainers, and metrics are in `common/`.

## Methods and required flags

| Method | Configuration | Required method-specific arguments |
| --- | --- | --- |
| `unet`, `convlstm`, `fno`, `neuralom`, `pde_transformer_mse` | Scratch | None |
| `dreamerv3` | RSSM forecasting adaptation | None |
| `lewm` | Factual LeWM adaptation | None |
| `eawm` | Event auxiliary adaptation | None; event weight is fixed at 0.1 |
| `tc_lewm` | Temporal-centered LeWM | None; temporal window is fixed at 4 |
| `climax` | Pretrained or scratch | `--variant pretrained --pretrained-path CHECKPOINT` or `--variant scratch` |
| `dpot`, `poseidon` | Pretrained or scratch | `--variant pretrained --pretrained-checkpoint CHECKPOINT` or `--variant scratch` |
| `pde_transformer_pretrained` | Pretrained | `--pretrained-checkpoint CHECKPOINT` |
| `enma` | VAE then flow | `--stage vae`, then `--stage flow --vae-checkpoint VAE_CHECKPOINT` |

The shared reader requires `DATA_ROOT/global_state.h5` with `state[372,5,90,180]`, `state_valid[372,90,180]`, coordinates, time, and cell area. It also requires `folds/main/manifest.json` with 12-month history, 3-month horizon, chronological splits, and training normalization, plus `folds/main/windows.csv` with split and origin indices. See the main [README](../README.md) for source products and preprocessing.

Formal runs use 30,000 updates per seed (17, 29, and 43), with full validation every 1,000 updates and best-checkpoint selection. `--max-validation-batches` is for smoke runs. Outputs are local JSON/JSONL artifacts.

## External code and checkpoints

Nine methods require upstream source files in a separately supplied ZIP. `common/external_files.json` lists required paths, including third-party notices where specified; the ZIP must contain each path under `code/`. This release hosts neither the ZIP nor an archive URL. Check upstream terms before redistribution. Supply a local archive, or a public HTTPS URL available without login and its expected SHA-256 digest:

```bash
python prepare_comparisons.py --archive LOCAL_ZIP
python prepare_comparisons.py --archive-url HTTPS_URL --sha256 EXPECTED_SHA256
```

The local ZIP digest is optional; a URL requires `--sha256`. Setup extracts only listed files into an external cache, which method entries verify. Use `--cache-root CACHE_ROOT` for setup and method commands to select a cache, or `--verify-only --cache-root CACHE_ROOT` to check it offline. Until the dependency ZIP is supplied, the nine methods cannot run from this release. Pretrained configurations also require their indicated model checkpoints; none are bundled here.

Install the root dependencies with `python -m pip install -r requirements.txt`. ENMA also requires `python -m pip install -r comparisons/requirements-enma.txt`, including PyTorch 2.6 or newer and `flow-matching`.

## Train and evaluate

For a single-stage method, replace `METHOD` and add its required flags from the table:

```bash
python comparisons/METHOD/train.py --release-root DATA_ROOT --output RUN_DIR --seed 17
python comparisons/METHOD/evaluate.py --release-root DATA_ROOT --checkpoint RUN_DIR/checkpoint_best.pt --split validation --output METRICS_JSON
```

For `climax`, `dpot`, and `poseidon`, pass the same `--variant` to evaluation. ENMA uses two training stages:

```bash
python comparisons/enma/train.py --stage vae --release-root DATA_ROOT --output VAE_RUN --seed 17
python comparisons/enma/train.py --stage flow --vae-checkpoint VAE_RUN/checkpoint_best.pt --release-root DATA_ROOT --output FLOW_RUN --seed 17
python comparisons/enma/evaluate.py --vae-checkpoint VAE_RUN/checkpoint_best.pt --flow-checkpoint FLOW_RUN/checkpoint_best.pt --release-root DATA_ROOT --split validation --output METRICS_JSON
```

Use `--split test` only after freezing method and checkpoint selection. Evaluation uses the specified checkpoint and does not select another one.
