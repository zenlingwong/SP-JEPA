#!/usr/bin/env python3
"""Export the consumed main-fold ocean data from an existing prepared release.

This is a data minimizer, not a source downloader or event-label producer. It
copies numeric arrays, rebuilds the reader's JSON/CSV contracts, and writes
only named scientific HDF5 attributes. The output path must not exist.
"""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np


STATE_DATASETS = ('state', 'state_valid', 'time', 'latitude', 'longitude',
                  'cell_area_km2', 'fgco2', 'fgco2_valid')
CARBON_DATASETS = ('values', 'valid', 'time', 'support', 'area_weights_km2')
EVENT_DATASETS = ('values', 'valid', 'time')
STATE_ATTRS = {
    'channel_names': json.dumps(['pco2', 'sst', 'sss', 'dic', 'alk']),
    'channel_units': json.dumps(['micro atm', 'degC', 'PSU', 'micro mol kg-1', 'micro mol kg-1']),
    'time_units': 'days since 1970-01-01',
    'state_valid_meaning': 'common five-channel native support pooled with >=0.5 full target-cell area coverage',
    'fgco2_units': 'molC m-2 yr-1; positive downward',
}
CARBON_NAMES = (
    'carbon_transition_fraction_lower', 'carbon_transition_fraction_upper',
    'carbon_sink_to_source_fraction_lower', 'carbon_sink_to_source_fraction_upper',
    'carbon_source_to_sink_fraction_lower', 'carbon_source_to_sink_fraction_upper',
    'carbon_known_support_fraction',
)
EVENT_CLOCK = 'monthly summaries of co-occurrence on the same UTC calendar day; not exact-instant simultaneity'
CARBON_SUPPORT = 'joint flux/state validity >=95% in this fold training months; outside support excluded'
WINDOW_COLUMNS = ('split', 'origin_index', 'history_start', 'history_end', 'target_start', 'target_end')
PRIMARY_PREFIXES = (
    'tc_sid_day_lower', 'tc_sid_day_upper', 'tc_sid_synoptic6h_max_lower',
    'tc_sid_synoptic6h_max_upper', 'mhw_component_count', 'mhw_active_point_count',
    'mhw_largest_component_points', 'cuti_component_count', 'cuti_active_band_count',
    'cuti_largest_component_bands',
)
SUMMARY_SUFFIXES = ('valid_day_count', 'daily_mean', 'daily_p90_linear', 'daily_max', 'days_ge2')
PRIMARY_NAMES = [f'{prefix}_{suffix}' for prefix in PRIMARY_PREFIXES for suffix in SUMMARY_SUFFIXES]
FAMILY_NAMES = [
    'family_cyclone_month_status', 'family_mhw_month_status', 'family_cuti_month_status',
    'family_same_utc_day_ge2_status', 'family_same_utc_day_ge2_days_lower',
    'family_same_utc_day_ge2_days_upper', 'family_same_utc_day_ge2_exposure_lower',
    'family_same_utc_day_ge2_exposure_upper', 'family_n_daily_max_lower',
    'family_n_daily_max_upper',
]
DIAGNOSTIC_NAMES = [f'{prefix}_{suffix}' for prefix in
                    ('diagnostic_source_object_sum_lower', 'diagnostic_source_object_sum_upper')
                    for suffix in SUMMARY_SUFFIXES]
CONTEXT_NAMES = ['enso_positive_context', 'enso_negative_context', 'enso_any_context']
FULL_NAMES = PRIMARY_NAMES + FAMILY_NAMES + DIAGNOSTIC_NAMES + CONTEXT_NAMES


def numeric_dataset(source, output, name):
    dataset = source[name]
    if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must be a numeric HDF5 dataset')
    if dataset.ndim == 0:
        output.create_dataset(name, data=dataset[()], track_times=False)
        return
    chunk = dataset.chunks or (1,) + dataset.shape[1:]
    target = output.create_dataset(name, dataset.shape, dtype=dataset.dtype, chunks=chunk,
                                   compression='gzip', compression_opts=1, track_times=False)
    for start in range(0, dataset.shape[0], chunk[0]):
        stop = min(start + chunk[0], dataset.shape[0])
        target[start:stop] = dataset[start:stop]


def copy_h5(source_path, output_path, names, attrs):
    with h5py.File(source_path, 'r') as source, h5py.File(output_path, 'x') as output:
        for name in names:
            numeric_dataset(source, output, name)
        for name, value in attrs(source).items():
            output.attrs[name] = value


def state_attrs(source):
    # Paths, history, author, and free-form source metadata are never copied.
    result = {}
    for name, expected in STATE_ATTRS.items():
        if name in source.attrs:
            value = source.attrs[name]
            if value != expected:
                raise ValueError(f'Unexpected {name} scientific attribute')
            result[name] = expected
    return result


def carbon_attrs(source):
    names = json.loads(source.attrs['target_names'])
    if names != list(CARBON_NAMES):
        raise ValueError('Unexpected carbon column names')
    if source.attrs.get('support') != CARBON_SUPPORT or source.attrs.get('time_support') != 'current and previous month only':
        raise ValueError('Unexpected carbon temporal or support contract')
    if 'encoder_visible' not in source.attrs or bool(source.attrs['encoder_visible']):
        raise ValueError('Carbon target must be retrospective')
    result = {'target_names': json.dumps(names), 'support': CARBON_SUPPORT,
              'time_support': 'current and previous month only', 'encoder_visible': False}
    if source.attrs.get('definition') == ('native adjacent 2-degree cell flux sign transitions; fractions of '
                                          'fold-admitted grid-cell area, not independent event counts'):
        result['definition'] = source.attrs['definition']
    return result


def event_attrs(source):
    names = json.loads(source.attrs['target_names'])
    if names != FULL_NAMES:
        raise ValueError('Event targets differ from the known source producer schema')
    expected_roles = {'primary_per_source_summary': PRIMARY_NAMES,
                      'legacy_family_diversity_alias': FAMILY_NAMES,
                      'diagnostic_heterogeneous_sum_not_primary': DIAGNOSTIC_NAMES,
                      'retrospective_context_not_causal_input': CONTEXT_NAMES}
    roles = json.loads(source.attrs['target_roles'])
    if not isinstance(roles, dict) or source['values'].shape[1] != len(names):
        raise ValueError('Event roles or columns do not match')
    if any(roles.get(key) != value for key, value in expected_roles.items()):
        raise ValueError('Event roles differ from the known source producer schema')
    if (source.attrs.get('clock') != EVENT_CLOCK or 'encoder_visible' not in source.attrs
            or bool(source.attrs['encoder_visible'])):
        raise ValueError('Unexpected event clock or visibility contract')
    if not bool(source.attrs.get('measurement_validity_is_not_online_authorization')):
        raise ValueError('Event validity must be retrospective only')
    result = {'target_names': json.dumps(names), 'target_roles': json.dumps(expected_roles),
              'clock': EVENT_CLOCK, 'encoder_visible': False,
              'measurement_validity_is_not_online_authorization': True}
    for name, expected in {'p90_method': 'numpy.quantile method=linear',
                           'primary_label': 'per-source count/extent vector',
                           'heterogeneous_sum_policy': 'diagnostic only; never a primary filter'}.items():
        if source.attrs.get(name) == expected:
            result[name] = expected
    return result


def fold_manifest(source):
    manifest = json.loads(source.read_text())
    if manifest.get('format') != 'global_state_fold_v1':
        raise ValueError('Expected global_state_fold_v1')
    result = {name: manifest[name] for name in ('format', 'history', 'horizon', 'state_mean', 'state_std', 'splits')}
    for name in ('counts', 'purged_counts'):
        if name in manifest:
            counts = manifest[name]
            if set(counts) != {'train', 'validation', 'test'} or not all(type(value) is int and value >= 0 for value in counts.values()):
                raise ValueError(f'Invalid {name}')
            result[name] = {split: counts[split] for split in ('train', 'validation', 'test')}
    if (result['history'], result['horizon']) != (12, 3):
        raise ValueError('Expected 12-month history and three-month target')
    if set(result['splits']) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, and test splits')
    for name in ('state_mean', 'state_std'):
        if len(result[name]) != 5 or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in result[name]):
            raise ValueError(f'Invalid {name}')
    for bounds in result['splits'].values():
        if len(bounds) != 2 or not all(isinstance(value, int) for value in bounds):
            raise ValueError('Invalid split bounds')
    return result


def copy_windows(source, output):
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'split', 'origin_index'}.issubset(reader.fieldnames or []):
            raise ValueError('Window CSV must have split and origin_index')
        columns = [name for name in WINDOW_COLUMNS if name in reader.fieldnames]
        rows = []
        for row in reader:
            if row['split'] not in {'train', 'validation', 'test'}:
                raise ValueError('Invalid window split')
            clean = {'split': row['split']}
            for name in columns[1:]:
                clean[name] = int(row[name])
            rows.append(clean)
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not source.is_dir():
        raise FileNotFoundError(source)
    source_fold = source / 'folds' / 'main'
    source_events = source / 'events'
    event_manifest = json.loads((source_events / 'manifest.json').read_text())
    if (event_manifest.get('format') != 'ocean_global_event_load_v2'
            or event_manifest.get('availability', {}).get('online_native_supervision_authorized') is not False
            or event_manifest.get('folds', {}).get('main', {}).get('event_values_encoder_visible') is not False):
        raise ValueError('Event package must declare retrospective-only main-fold labels')
    fold = fold_manifest(source_fold / 'manifest.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ocean-export-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'release'
        (stage / 'folds' / 'main').mkdir(parents=True)
        (stage / 'events' / 'main').mkdir(parents=True)
        copy_h5(source / 'global_state.h5', stage / 'global_state.h5', STATE_DATASETS, state_attrs)
        copy_h5(source_fold / 'carbon_monthly.h5', stage / 'folds' / 'main' / 'carbon_monthly.h5',
                CARBON_DATASETS, carbon_attrs)
        copy_h5(source_events / 'main' / 'monthly_events.h5', stage / 'events' / 'main' / 'monthly_events.h5',
                EVENT_DATASETS, event_attrs)
        with h5py.File(stage / 'global_state.h5') as state, \
                h5py.File(stage / 'folds' / 'main' / 'carbon_monthly.h5') as carbon, \
                h5py.File(stage / 'events' / 'main' / 'monthly_events.h5') as events:
            if not np.array_equal(state['time'][:], carbon['time'][:]) or not np.array_equal(state['time'][:], events['time'][:]):
                raise ValueError('State, carbon, and event clocks differ')
        (stage / 'folds' / 'main' / 'manifest.json').write_text(json.dumps(fold, indent=2) + '\n')
        copy_windows(source_fold / 'windows.csv', stage / 'folds' / 'main' / 'windows.csv')
        if (source_fold / 'windows_purged.csv').exists():
            copy_windows(source_fold / 'windows_purged.csv', stage / 'folds' / 'main' / 'windows_purged.csv')
        clean_events = {'format': 'ocean_global_event_load_v2',
                        'availability': {'online_native_supervision_authorized': False},
                        'folds': {'main': {'monthly_events_h5': 'main/monthly_events.h5',
                                           'event_values_encoder_visible': False}}}
        (stage / 'events' / 'manifest.json').write_text(json.dumps(clean_events, indent=2) + '\n')
        stage.rename(output)
    print(json.dumps({'output': str(output), 'fold': 'main',
                      'files': sum(path.is_file() for path in output.rglob('*'))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == '__main__':
    main()
