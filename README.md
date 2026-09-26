# Learning Editable Predictive States for Ocean World Modeling

State-Perturbation Joint-Embedding Predictive Architecture (SP-JEPA) represents each frame with 32 fixed physical coordinates and 128 learned coordinates. A causal Transformer predicts coordinate increments and the next learned state. On the **Global Ocean Carbon–Physics dataset** (the ocean dataset), the model predicts three months from 12 monthly observations. In the **Paired Transport–Exchange environment** (the paired environment), the model predicts six steps from six observations using paired natural and perturbed trajectories. The two environments use separately trained parameters.

## Layout

```text
main/
  model/                 Current model and historical-event fusion
  src/                   Data readers, simulator, losses, and metrics
  prepare_ocean.py        Training-only event normalization
  train_ocean.py          Two-stage ocean training
  evaluate_ocean.py       Frozen-checkpoint ocean evaluation
  prepare_paired.py       Paired simulation data
  train_paired.py         Natural and paired training
  evaluate_paired.py      Response and calibration evaluation
comparisons/
  <method>/              Method-specific train and evaluate entries
  common/                Shared adapters, trainers, and metrics
data_preparation/        State aggregation, carbon targets, and anonymous export
prepare_comparisons.py    Import a supplied dependency ZIP into an external cache
requirements.txt         Main-model Python dependencies
```

Use Python 3.10 or newer. Install the main dependencies with `python -m pip install -r requirements.txt`. Run the commands below from the package root. Comparison instructions and additional requirements are in [comparisons/README.md](comparisons/README.md).

## Global Ocean Carbon–Physics dataset

The prepared data release contains:

```text
data/ocean/
  global_state.h5
  folds/main/manifest.json
  folds/main/windows.csv
  folds/main/carbon_monthly.h5
  events/manifest.json
  events/main/monthly_events.h5
```

The field file has 372 monthly frames, five variables, and a 90-by-180 grid. It also contains validity masks, area, coordinates, time, and carbon-flux fields for the event targets. Fold manifests supply training-only normalization and chronological split boundaries. The default fold uses unpurged windows.

### Sources

The retained input metadata and processing code identify the following public products.

| Quantity | Product and native variable | Source reference |
| --- | --- | --- |
| pCO2, DIC, ALK | Surface ocean carbon product `MULTIOBS_GLO_BGC_CARBON_SURFACE_MYNRT_015_008`; `spco2`, `tco2`, `talk` | [Product DOI](https://doi.org/10.48670/moi-00047) |
| Carbon-flux diagnostic | The same product, `fgco2`; positive downward, in molC m-2 yr-1 | [Product documentation](https://documentation.marine.copernicus.eu/PUM/CMEMS-MOB-PUM-015-008.pdf) |
| SST | OISST 2.1, supplied monthly `sst.mon.mean.nc`, variable `sst`; version verified in its file attributes | [Monthly product catalogue](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html) |
| SSS | ORAS5 v0.1 control monthly files, `vosaline` at the first depth level (about 0.50576 m); consolidated and operational files | [Product DOI](https://doi.org/10.24381/cds.67e8eeb7) |
| Tropical-cyclone catalogue | IBTrACS v04r01, main tracks, using SID, time, nature and wind records | [Dataset DOI](https://doi.org/10.25921/82ty-9e16) |
| Coastal upwelling index | Daily CUTI over 17 latitude bands from 31 to 47 degrees north | [Data endpoint](https://oceanview.pfeg.noaa.gov/erddap/info/erdCUTIdaily/index.html) |

All model windows cover January 1993 through December 2023. The carbon fields are retrospective reconstructions; DIC and carbon flux share upstream derivations with other carbon variables. The task is forecasting these reconstructed products. Product DOI/catalogue identifiers do not freeze a historical download: the retained records do not establish an exact carbon-product release revision or CUTI snapshot version. A shared processed release should therefore carry its own file checksums and source-snapshot record.

### Processing

1. Match calendar months and normalize longitude to `[-180,180)`. The aligned native field has a 713-by-1440 grid at 0.25 degrees. Salinity was mapped from adjacent native-grid triangles in three-dimensional spherical coordinates, requiring all three source vertices to be valid and using no extrapolation. The global-state builder replaces the intermediate SST with values selected at exact original SST nodes and months.
2. Form the common validity mask of all five fields. Aggregate native cells using their spherical areas to a regular 2-degree grid with centers from -89 to 89 degrees latitude and -179 to 179 degrees longitude. Accept a coarse cell only when valid native area reaches half its full spherical cell area. Store invalid values as zero with a separate mask. Pool carbon flux on the common field-and-flux support.
3. Fit each channel's mean and population standard deviation from all valid coarse training pixels, with equal pixel weight in these normalization statistics. Spatial losses separately use cell-area weights. Keep the chronological boundaries and all legal rolling windows below.
4. Construct source event observations. Cyclone daily lower counts use distinct confirmed SIDs; upper counts also admit catalogue ambiguity. Monthly counts are bounded by `n/(1+n)`. CUTI active bands exceed the positive training-calendar-month 90th percentile; their complete-month daily mean count is divided by 17. Unknown source values retain explicit known masks.
5. Derive monthly SST warm-area fraction from training-calendar-month 90th-percentile thresholds. Derive trailing three-month regional temperature classes from the same training-calibrated SST record. Derive carbon sign-change and sink-area bounds on support valid in at least 95 percent of training months. Lower and upper endpoints describe identification uncertainty, not prediction confidence intervals.
6. Fit event climatology and anomaly scales on training-origin months. Only event observations at or before the forecast origin enter the model, and the retrospective cyclone upper bound is excluded from its input. Future event values are supervision targets.

| Split | Half-open month indices | Split month range | Rolling origins |
| --- | --- | --- | ---: |
| Train | `[0,297)` | 1993-01 to 2017-09 | 283 |
| Validation | `[297,334)` | 2017-10 to 2020-10 | 35 |
| Test | `[334,372)` | 2020-11 to 2023-12 | 36 |

Each origin uses 12 history months and the next three target months. Rolling evaluation can use already observed history from the preceding split. The processed event table also contains older auxiliary summaries; the current model consumes the source cyclone/upwelling columns and computes its other event indicators through `main/src/event_state_data.py`.

### Preparation scripts and inputs

| Script | Input | Output |
| --- | --- | --- |
| `data_preparation/build_global_state.py` | Already aligned native field/mask archive, original monthly SST file, and a JSON manifest of 372 carbon files | Global state, flux/transition arrays, main-fold normalization and windows |
| `data_preparation/build_carbon_targets.py` | Global state, flux-validity and sign-transition arrays | Seven monthly carbon-bound/support columns |
| `main/src/event_state_data.py` | Prepared state, monthly source-event table and carbon table | Training-calibrated event bounds and known masks |
| `main/prepare_ocean.py` | Prepared main-fold release | Event means and anomaly scales |
| `data_preparation/export_ocean.py` | Existing complete prepared release | Minimal reader-compatible copy with only scientific metadata |
| `main/prepare_paired.py` | Seed and the included simulator | Independent synthetic paired data |

The state builder expects native datasets `field[372,5,713,1440]`, `valid_mask` of the same shape, `latitude`, `longitude`, and `time` in days since 1970-01-01. Its carbon manifest contains a `carbon_files` array in monthly order; relative paths resolve from that manifest's directory. These scripts start from aligned/native inputs and an independently prepared source-event table. The upstream alignment producer and full catalogue-processing pipeline are not included, so the package does not provide a one-command rebuild from public raw downloads.

```bash
python data_preparation/build_global_state.py --base inputs/native_fields.nc --sst inputs/sst.mon.mean.nc --carbon-manifest inputs/carbon_files.json --output data/ocean
python data_preparation/build_carbon_targets.py --root data/ocean
```

Before model training, also supply `events/manifest.json` and `events/main/monthly_events.h5` from the complete prepared release. The model-facing event reader validates their time, target-column and retrospective-availability contracts.

Fit the 12-by-13 event climatology and 13 anomaly scales on training-origin months:

```bash
python main/prepare_ocean.py --data-root data/ocean --output data/ocean_event_statistics.json
```

Run the main recipe independently for seeds 17, 29, and 43:

```bash
python main/train_ocean.py --recipe main --data-root data/ocean --event-statistics data/ocean_event_statistics.json --seed 17 --device cuda:0 --output runs/ocean_s17 --save-predictions
python main/evaluate_ocean.py --checkpoint runs/ocean_s17/stage2_best.pt --data-root data/ocean --split validation --device cuda:0 --output runs/ocean_s17/evaluation.json
```

The main recipe uses 28,000 first-stage updates and 2,000 second-stage updates, with validation every 500 and 100 updates. Each stage selects the checkpoint with the lowest validation field MSE. Both stages use batches of eight non-overlapping windows and a fresh optimizer and sampler. The second stage fuses historical event observations into the initial state. All six event families are supervised; the reported inference-gated output suppresses three seasonal residuals.

`--recipe development` selects the 3,000-plus-2,000 development recipe with fixed endpoints. Checkpoints contain model weights, the physical basis, event statistics, field climatology, and model settings. Datasets and pretrained weights are supplied separately.

## Paired Transport–Exchange environment

```bash
python main/prepare_paired.py --seed 17 --output data/paired_s17.pt
python main/train_paired.py --data data/paired_s17.pt --device cuda:0 --output runs/paired_s17
python main/evaluate_paired.py --data data/paired_s17.pt --checkpoint runs/paired_s17/best.pt --device cuda:0 --output runs/paired_s17/evaluation.json
```

Preparation generates 512 training origins, 288 pair-actions with 576 changed trajectories, and 1,024 rows in each independent selection and calibration bank. Training uses 1,000 natural updates followed by 5,000 paired updates. Checkpoint selection uses two-branch field MSE.

Evaluation reports RS, C, P, LCS, 2,000-resample mechanism intervals, response resolution, and field capture and endpoint calibration. `warmup.pt` is a pipeline intermediate. Event and joint qualification are outside this field evaluator.

## Dependencies and licensing

Third-party model code and its required notices are imported from a separately supplied ZIP into a cache outside this package. The dependency preparation command is described in [comparisons/README.md](comparisons/README.md). No private source repository or author-linked revision is embedded. An external dependency bundle must actually be made available to recipients; this package does not publish one. No project-wide license was declared for the first-party source, and this snapshot does not assign a new one. Data sources retain their own citation and redistribution terms.

## Providing processed data for review

The [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) allow supplementary code and anonymous source links, and recommend documenting dataset processing in supplementary material. The materials and linked pages must preserve author anonymity. The guidance also warns against hosts that track reviewers.

For this task, provide the code ZIP as supplementary material and the processed dataset as a separate anonymous download when it exceeds the submission form's file limit. The currently required prepared files total about 79 MB before a new export; the form's actual limit should be checked rather than assumed. The data download should be available without requesting access from an author or exposing an account profile. Include relative filenames, shapes, units, split definitions, preprocessing commands, checksums, and source citations. A licence check for redistribution is still required for the constituent products.

Create the minimal sharing copy from an existing prepared release:

```bash
python data_preparation/export_ocean.py --source inputs/prepared_ocean --output data/ocean_share
python main/prepare_ocean.py --data-root data/ocean_share --output data/ocean_event_statistics.json
```

The exporter reconstructs the necessary JSON/CSV metadata and copies numeric HDF5 arrays without original paths, authors, free-form history, or source-manifest metadata. Inspect the final archive and its download page before submission. No processed data or hosted data link is bundled with this code snapshot.
