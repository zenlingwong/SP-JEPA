# Comparison methods

This directory contains 14 comparison methods and 17 training configurations for the three-month global forecast task. Each method directory has a `train.py` and `evaluate.py` entry. Shared data reading, trainers, metrics, and adapters are in `common/`.

| Method directory | Configuration | Additional training arguments |
| --- | --- | --- |
| `unet`, `convlstm`, `fno`, `neuralom` | One scratch configuration each | None |
| `pde_transformer_mse` | Scratch | None |
| `dreamerv3` | RSSM forecasting adaptation | None |
| `lewm` | Factual LeWM adaptation | None |
| `eawm` | Event auxiliary adaptation | Event weight is fixed to 0.1 by the entry |
| `tc_lewm` | Temporal centered LeWM | Temporal window is fixed to 4 by the entry |
| `climax` | Pretrained and scratch | `--variant pretrained --pretrained-path CHECKPOINT` or `--variant scratch` |
| `dpot`, `poseidon` | Pretrained and scratch each | `--variant pretrained --pretrained-checkpoint CHECKPOINT` or `--variant scratch` |
| `pde_transformer_pretrained` | Pretrained | `--pretrained-checkpoint CHECKPOINT` |
| `enma` | VAE followed by flow | `--stage vae`, then `--stage flow --vae-checkpoint VAE_CHECKPOINT` |

The state reader expects `global_state.h5`, `folds/main/manifest.json`, and `folds/main/windows.csv` under `DATA_ROOT`: 372 monthly five-channel frames on a 90×180 grid, with 12 visible months and three forecast months. Formal training uses 30,000 optimizer updates per seed for seeds 17, 29, and 43. Full validation runs every 1,000 updates by default and selects `checkpoint_best.pt`; `--max-validation-batches` is for smoke runs only. Training and evaluation write local JSON/JSONL artifacts.

Nine methods require model source files and mandatory license notices supplied separately as a dependency ZIP. The package lists the required relative paths in `common/external_files.json`; it contains neither the dependency files nor an archive URL. The ZIP must contain each listed file under `code/`. Provision it only where the source licenses and access terms allow. Supply a local ZIP outside this package, or an anonymously accessible HTTPS URL with an expected SHA-256 digest:

```bash
python prepare_comparisons.py --archive LOCAL_ZIP
python prepare_comparisons.py --archive-url ANON_URL --sha256 EXPECTED_ARCHIVE_DIGEST
```

For a local ZIP, `--sha256` is optional; if supplied, setup checks it before extraction. A URL requires `--sha256` and must be available without login. Setup extracts only the listed model files and notices into `~/.cache/environment-model/comparisons/code/`, outside this package. It records the archive digest and per-file digests in `~/.cache/environment-model/comparisons/receipt.json`; method entries verify their files against that external receipt.

No repository checkout, credentials, or URL are stored. Use `--cache-root CACHE_ROOT` with setup and a method entry to choose another external cache. Run `python prepare_comparisons.py --verify-only --cache-root CACHE_ROOT` to check it offline. Until an accessible dependency ZIP is supplied, those nine methods cannot be independently run from this release.

Install the packages in `requirements.txt` in a suitable environment before running the methods. ENMA additionally requires `requirements-enma.txt`, including a PyTorch build with `torch.nn.attention.flex_attention` and `flow_matching`. Pretrained configurations require their indicated checkpoints. No checkpoint is bundled here.

For any single-stage method, train and evaluate a frozen checkpoint as follows, replacing `METHOD` and adding the method's arguments from the table:

```bash
python comparisons/METHOD/train.py --release-root DATA_ROOT --output RUN_DIR --seed 17
python comparisons/METHOD/evaluate.py --release-root DATA_ROOT --checkpoint RUN_DIR/checkpoint_best.pt --split validation --output METRICS_JSON
```

For `climax`, `dpot`, and `poseidon`, pass the matching `--variant` to evaluation too. The two-stage ENMA sequence is:

```bash
python comparisons/enma/train.py --stage vae --release-root DATA_ROOT --output VAE_RUN --seed 17
python comparisons/enma/train.py --stage flow --vae-checkpoint VAE_RUN/checkpoint_best.pt --release-root DATA_ROOT --output FLOW_RUN --seed 17
python comparisons/enma/evaluate.py --vae-checkpoint VAE_RUN/checkpoint_best.pt --flow-checkpoint FLOW_RUN/checkpoint_best.pt --release-root DATA_ROOT --split validation --output METRICS_JSON
```

Use `--split test` only after method and checkpoint selection are frozen. Evaluation uses the specified checkpoint and does not select another one.
