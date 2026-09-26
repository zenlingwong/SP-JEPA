#!/usr/bin/env python3
"""Build the main-fold 2-degree state from already aligned native inputs.

The native five-channel base is an input to this script; its upstream producer
is not included here. The carbon manifest is JSON with a ``carbon_files`` list
of 372 monthly CMEMS HDF5 paths, in calendar order. Relative entries resolve
from the manifest directory.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


EARTH_RADIUS_KM = 6371.0
CHANNELS = ['pco2', 'sst', 'sss', 'dic', 'alk']
UNITS = ['micro atm', 'degC', 'PSU', 'micro mol kg-1', 'micro mol kg-1']
SPLITS = {'train': (0, 297), 'validation': (297, 334), 'test': (334, 372)}


def geometry(latitude, longitude):
    lat = np.asarray(latitude, dtype='float64')
    lon = np.asarray(longitude, dtype='float64')
    if lat.shape != (713,) or lon.shape != (1440,):
        raise ValueError('Expected 713x1440 native coordinates')
    if not np.allclose(np.diff(lat), .25) or not np.allclose(np.diff(lon), .25):
        raise ValueError('Expected 0.25-degree native grid')
    row = np.floor((lat + 90) / 2).astype(int)
    column = np.floor((lon + 180) / 2).astype(int)
    if not np.all((0 <= row) & (row < 90)) or not np.all((0 <= column) & (column < 180)):
        raise ValueError('Native centers lie outside target grid')
    edges = np.concatenate(([lat[0] - .125], (lat[:-1] + lat[1:]) / 2, [lat[-1] + .125]))
    native_area_row = EARTH_RADIUS_KM ** 2 * np.deg2rad(.25) * (
        np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1])))
    target_edges = np.arange(-90, 92, 2, dtype='float64')
    target_area_row = EARTH_RADIUS_KM ** 2 * np.deg2rad(2) * (
        np.sin(np.deg2rad(target_edges[1:])) - np.sin(np.deg2rad(target_edges[:-1])))
    full_area = target_area_row[:, None] * np.ones((1, 180), dtype='float64')
    domain_area = np.zeros((90, 180), dtype='float64')
    for target_row in range(90):
        domain_area[target_row] = native_area_row[row == target_row].sum() * 8
    fraction = domain_area / full_area
    if not np.allclose(fraction[1:], 1) or not (0 < fraction[0, 0] < 1):
        raise ValueError('Unexpected retained native-domain geometry')
    return row, column, native_area_row, full_area, fraction.astype('float32')


def pool(values, valid, row, native_area_row, full_area):
    """Area-weight native cells and admit >=50% of each full target cell."""
    values = np.asarray(values, dtype='float64')
    valid = np.asarray(valid, bool)
    scalar = values.ndim == 2
    if scalar:
        values = values[None]
    if values.shape[1:] != (713, 1440) or valid.shape != (713, 1440):
        raise ValueError('Unexpected native frame shape')
    area = native_area_row[:, None]
    weighted = np.where(valid[None], values * area[None], 0).reshape(values.shape[0], 713, 180, 8).sum(3)
    denominator = (valid * area).reshape(713, 180, 8).sum(2)
    numerator = np.zeros((values.shape[0], 90, 180), dtype='float64')
    present = np.zeros((90, 180), dtype='float64')
    for target_row in range(90):
        source_rows = row == target_row
        numerator[:, target_row] = weighted[:, source_rows].sum(1)
        present[target_row] = denominator[source_rows].sum(0)
    coverage = present / full_area
    accepted = coverage >= .5
    pooled = np.divide(numerator, present[None], out=np.zeros_like(numerator), where=present[None] > 0)
    pooled[:, ~accepted] = 0
    return (pooled[0] if scalar else pooled).astype('float32'), accepted, coverage.astype('float32')


def month_lookup(raw_time, epoch):
    months = (np.datetime64(epoch) + np.rint(raw_time).astype('timedelta64[D]')).astype('datetime64[M]')
    return {str(month): index for index, month in enumerate(months)}


def write_windows(path, purged=False):
    rows = []
    for split, (lo, hi) in SPLITS.items():
        first = max(11, lo - 1)
        if purged:
            first = max(first, lo + 11)
        for origin in range(first, hi - 3):
            rows.append({'split': split, 'origin_index': origin, 'history_start': origin - 11,
                         'history_end': origin, 'target_start': origin + 1, 'target_end': origin + 3})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {split: sum(row['split'] == split for row in rows) for split in SPLITS}


def build(base, sst, carbon_manifest, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    carbon_entries = json.loads(carbon_manifest.read_text())['carbon_files']
    carbon_files = [Path(entry) if Path(entry).is_absolute() else carbon_manifest.parent / entry
                    for entry in carbon_entries]
    if len(carbon_files) != 372 or not all(path.is_file() for path in carbon_files):
        raise ValueError('Need 372 existing monthly carbon files')
    output.mkdir(parents=True)
    with h5py.File(base) as native, h5py.File(sst) as noaa, h5py.File(output / 'global_state.h5', 'x') as out:
        lat, lon, time = native['latitude'][:], native['longitude'][:], native['time'][:]
        row, column, native_area_row, full_area, fraction = geometry(lat, lon)
        dates = (np.datetime64('1970-01-01') + time.astype('timedelta64[D]')).astype('datetime64[M]')
        if not np.array_equal(dates, np.arange('1993-01', '2024-01', dtype='datetime64[M]')):
            raise ValueError('Expected continuous 1993-01 through 2023-12 native clock')
        noaa_lat = noaa['lat'][:]
        noaa_lon = (noaa['lon'][:] + 180) % 360 - 180
        noaa_order = np.argsort(noaa_lon)
        noaa_rows = np.flatnonzero(np.isin(noaa_lat, lat))
        noaa_time = month_lookup(noaa['time'][:], '1800-01-01')
        if not (np.array_equal(noaa_lat[noaa_rows], lat) and np.array_equal(noaa_lon[noaa_order], lon)
                and all(str(month) in noaa_time for month in dates)):
            raise ValueError('NOAA SST cannot be selected on exact native nodes and months')

        frame = {'state': ((372, 5, 90, 180), 'f4', (1, 5, 90, 180)),
                 'state_valid': ((372, 90, 180), '?', (1, 90, 180)),
                 'native_valid_area_fraction': ((372, 90, 180), 'f4', (1, 90, 180)),
                 'fgco2': ((372, 90, 180), 'f4', (1, 90, 180)),
                 'fgco2_valid': ((372, 90, 180), '?', (1, 90, 180)),
                 'sign_transition': ((372, 90, 180), 'i1', (1, 90, 180)),
                 'transition_valid': ((372, 90, 180), '?', (1, 90, 180))}
        datasets = {name: out.create_dataset(name, shape, dtype, chunks=chunks, compression='gzip', compression_opts=1)
                    for name, (shape, dtype, chunks) in frame.items()}
        out['time'] = time
        out['latitude'] = np.arange(-89, 90, 2, dtype='float32')
        out['longitude'] = np.arange(-179, 180, 2, dtype='float32')
        out['cell_area_km2'] = full_area
        out['native_domain_fraction'] = fraction
        stats = np.zeros((3, 5), dtype='float64')
        previous_flux = previous_valid = None
        for t, month in enumerate(dates):
            fields = native['field'][t].astype('float64')
            mask = native['valid_mask'][t].astype(bool)
            direct_sst = noaa['sst'][noaa_time[str(month)], noaa_rows][:, noaa_order]
            sst_valid = np.isfinite(direct_sst) & (np.abs(direct_sst) < 1e30)
            fields[1] = direct_sst
            common = mask[[0, 2, 3, 4]].all(0) & sst_valid
            pooled, state_valid, coverage = pool(fields, common, row, native_area_row, full_area)
            datasets['state'][t] = pooled
            datasets['state_valid'][t] = state_valid
            datasets['native_valid_area_fraction'][t] = coverage
            with h5py.File(carbon_files[t]) as carbon:
                carbon_lon = (carbon['longitude'][:] + 180) % 360 - 180
                order = np.argsort(carbon_lon)
                carbon_month = (np.datetime64('1950-01-01T00') +
                                np.rint(carbon['time'][:]).astype('timedelta64[h]')).astype('datetime64[M]')
                if (carbon['time'].shape != (1,) or not np.array_equal(carbon['latitude'][:], lat)
                        or not np.array_equal(carbon_lon[order], lon) or carbon_month[0] != month):
                    raise ValueError(f'Carbon grid or month mismatch at index {t}')
                flux = carbon['fgco2'][0][:, order]
                flux_valid_native = common & np.isfinite(flux) & (np.abs(flux) < 1e30)
            pooled_flux, flux_valid, _ = pool(flux, flux_valid_native, row, native_area_row, full_area)
            datasets['fgco2'][t] = pooled_flux
            datasets['fgco2_valid'][t] = flux_valid
            transition_valid = np.zeros((90, 180), bool)
            transition = np.zeros((90, 180), np.int8)
            if previous_flux is not None:
                transition_valid = flux_valid & previous_valid
                transition[transition_valid & (previous_flux > 0) & (pooled_flux < 0)] = 1
                transition[transition_valid & (previous_flux < 0) & (pooled_flux > 0)] = -1
            datasets['sign_transition'][t] = transition
            datasets['transition_valid'][t] = transition_valid
            previous_flux, previous_valid = pooled_flux, flux_valid
            if t < SPLITS['train'][1]:
                for channel in range(5):
                    values = pooled[channel, state_valid].astype('float64')
                    stats[0, channel] += len(values)
                    stats[1, channel] += values.sum()
                    stats[2, channel] += np.dot(values, values)
        for key, value in {'channel_names': json.dumps(CHANNELS), 'channel_units': json.dumps(UNITS),
                           'time_units': 'days since 1970-01-01',
                           'state_valid_meaning': 'common five-channel native support pooled with >=0.5 full target-cell area coverage',
                           'fgco2_units': 'molC m-2 yr-1; positive downward',
                           'transition_codes': '+1 sink-to-source, -1 source-to-sink, 0 no observed sign change'}.items():
            out.attrs[key] = value

    count, total, squares = stats
    mean = total / count
    std = np.sqrt(squares / count - mean ** 2)
    folder = output / 'folds' / 'main'
    folder.mkdir(parents=True)
    counts = write_windows(folder / 'windows.csv')
    purged_counts = write_windows(folder / 'windows_purged.csv', purged=True)
    manifest = {'format': 'global_state_fold_v1', 'fold': 'main',
                'splits': {name: list(bounds) for name, bounds in SPLITS.items()},
                'history': 12, 'horizon': 3, 'state_mean': mean.tolist(), 'state_std': std.tolist(),
                'normalization': 'unweighted all valid coarse pixels in main training interval only',
                'counts': counts, 'purged_counts': purged_counts}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'counts': counts, 'purged_counts': purged_counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--sst', type=Path, required=True)
    parser.add_argument('--carbon-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.sst, args.carbon_manifest, args.output)


if __name__ == '__main__':
    main()
