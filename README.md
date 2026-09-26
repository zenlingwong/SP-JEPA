# Learning Editable Predictive States for Ocean World Modeling

State-Perturbation Joint-Embedding Predictive Architecture (SP-JEPA) represents each frame with 32 fixed physical coordinates and 128 learned coordinates. A causal Transformer predicts coordinate increments and the next learned state. The ocean task forecasts three months from 12 monthly global fields; the paired transport–exchange task forecasts six steps from six observations. The tasks use separately trained parameters.

## Layout and environment

```text
main/                  Model, data readers, training and evaluation
data_preparation/      Global-state and carbon-target preparation
comparisons/           Baseline and method-specific entries
prepare_comparisons.py External comparison-dependency setup
requirements.txt       Main-model dependencies
```

Use Python 3.10 or newer. From the package root, install the main dependencies with `python -m pip install -r requirements.txt`. Comparison-specific setup is in [comparisons/README.md](comparisons/README.md).

## Global Ocean Carbon–Physics data

The code expects a separately supplied prepared release with this layout:

```text
data/
  global_state.h5
  folds/main/manifest.json
  folds/main/windows.csv
  folds/main/carbon_monthly.h5
  events/manifest.json
  events/main/monthly_events.h5
```

`global_state.h5` contains 372 monthly frames from January 1993 through December 2023 on a 90×180 grid: `state[372,5,90,180]`, `state_valid[372,90,180]`, coordinates, time, cell area, carbon flux, and transition fields. Channels are pCO₂, SST, SSS, DIC, and ALK. Fold metadata supplies training-only normalization and rolling windows. The event package uses format `ocean_global_event_load_v2`; its HDF5 table contains aligned monthly `time`, `values[T,N]`, and same-shaped `valid` arrays with target metadata. The reader checks clock alignment and retrospective-only availability.

### Source products

| Quantity | Product and native variable | Reference |
| --- | --- | --- |
| pCO₂, DIC, ALK | `MULTIOBS_GLO_BGC_CARBON_SURFACE_MYNRT_015_008`; `spco2`, `tco2`, `talk` | [Product DOI](https://doi.org/10.48670/moi-00047) |
| Carbon flux | Same product, `fgco2`; positive downward, molC m⁻² yr⁻¹ | [Product documentation](https://documentation.marine.copernicus.eu/PUM/CMEMS-MOB-PUM-015-008.pdf) |
| SST | OISST 2.1, monthly `sst.mon.mean.nc`, `sst` | [Monthly product catalogue](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html) |
| SSS | ORAS5 v0.1 control monthly files, `vosaline` at the first depth level (~0.50576 m) | [Product DOI](https://doi.org/10.24381/cds.67e8eeb7) |
| Tropical cyclones | IBTrACS v04r01 main tracks: SID, time, nature, and wind | [Dataset DOI](https://doi.org/10.25921/82ty-9e16) |
| Coastal upwelling | Daily CUTI over 17 latitude bands, 31–47°N | [Data endpoint](https://oceanview.pfeg.noaa.gov/erddap/info/erdCUTIdaily/index.html) |

Carbon fields are retrospective reconstructions, and DIC and `fgco2` share upstream derivations with other carbon variables. The retained records do not identify an exact carbon-product release revision or CUTI snapshot; product links alone do not freeze those downloads.

### Processing and splits

The aligned native field input is `field[372,5,713,1440]` with a same-shaped `valid_mask`, 713 latitudes, 1440 longitudes, and monthly time in days since 1970-01-01. The builder uses a 0.25° grid, matches calendar months, and normalizes longitude to `[-180,180)`. It maps salinity from adjacent valid native-grid triangles in 3D spherical coordinates without extrapolation, then selects SST at exact source nodes and months. It pools the common five-field support by spherical cell area onto a 2° grid, requiring at least half of each target cell's full area; invalid values are zero with a separate mask. Carbon flux uses common field-and-flux support.

Channel normalization uses valid coarse training pixels with equal pixel weight; spatial losses use cell-area weights. Event thresholds, climatology, and anomaly scales are fitted on training data only. Only event observations available by a forecast origin enter the model; future event values are supervision targets.

| Split | Half-open indices | Months | Rolling origins |
| --- | --- | --- | ---: |
| Train | `[0,297)` | 1993-01–2017-09 | 283 |
| Validation | `[297,334)` | 2017-10–2020-10 | 35 |
| Test | `[334,372)` | 2020-11–2023-12 | 36 |

Each origin uses 12 history months and the next three target months. Rolling evaluation may use observed history from the preceding split.

### Prepare and run

`build_global_state.py` requires an aligned native archive with `field[372,5,713,1440]`, a same-shaped `valid_mask`, `latitude[713]`, `longitude[1440]`, and `time[372]`; an SST HDF5 file with `lat`, `lon`, `time`, and `sst`; and a JSON manifest with a calendar-ordered `carbon_files` list of 372 HDF5 files containing `latitude`, `longitude`, `time`, and `fgco2`. Relative carbon paths resolve from the manifest directory. The upstream alignment and catalogue-processing pipeline is not included. A separately prepared event table is also required and is not generated here, so this package does not rebuild the release from raw public downloads.

```bash
python data_preparation/build_global_state.py --base inputs/native_fields.nc --sst inputs/sst.mon.mean.nc --carbon-manifest inputs/carbon_files.json --output data
python data_preparation/build_carbon_targets.py --root data
```

Supply `events/manifest.json` and `events/main/monthly_events.h5` from the complete prepared release before fitting event statistics:

```bash
python main/prepare_ocean.py --data-root data --output data/event_statistics.json
```

The main recipe runs 28,000 first-stage and 2,000 second-stage updates, validates every 500 and 100 updates, and selects each stage by validation field MSE. Run the formal seeds 17, 29, and 43 independently; this command shows seed 17:

```bash
python main/train_ocean.py --recipe main --data-root data --event-statistics data/event_statistics.json --seed 17 --device cuda:0 --output runs/ocean_s17 --save-predictions
python main/evaluate_ocean.py --checkpoint runs/ocean_s17/stage2_best.pt --data-root data --split validation --device cuda:0 --output runs/ocean_s17/evaluation.json
```

`--recipe development` selects the 3,000+2,000-update fixed-endpoint recipe. Checkpoints include model weights, the physical basis, event statistics, field climatology, and model settings; datasets and pretrained comparison weights are separate inputs.

## Paired transport–exchange task

The included simulator prepares synthetic paired data:

```bash
python main/prepare_paired.py --seed 17 --output data/paired_s17.pt
python main/train_paired.py --data data/paired_s17.pt --device cuda:0 --output runs/paired_s17
python main/evaluate_paired.py --data data/paired_s17.pt --checkpoint runs/paired_s17/best.pt --device cuda:0 --output runs/paired_s17/evaluation.json
```

Preparation creates 512 training origins, 288 pair-actions with 576 changed trajectories, and 1,024 rows in each selection and calibration bank. Training uses 1,000 natural updates followed by 5,000 paired updates. Evaluation reports RS, C, P, LCS, 2,000-resample mechanism intervals, and field response and calibration metrics.

## Licensing

No project-wide license is declared for first-party source. Comparison source files and required third-party notices must be supplied separately as described in [comparisons/README.md](comparisons/README.md); no dependency archive or pretrained checkpoint is hosted here. Data products retain their own citation and redistribution terms.
