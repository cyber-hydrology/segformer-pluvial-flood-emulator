#!/usr/bin/env python3
"""
Rainfall preprocessing for the urban pluvial flood emulator.

Converts rainfall CSV files (10 min rainfall depth in mm) into the normalised
rainfall features used by the released models.

Rainfall features (one row per 10 min step)
  0  log1p(rainfall)
  1  log1p(1 h trailing moving sum)
  2  log1p(6 h trailing moving sum)
  3  log1p(24 h trailing moving sum)
  4  log1p(cumulative rainfall since event start)
  5  temporal gradient of log1p(rainfall)
Each feature is divided by its 99th percentile over the 90 training storms
(absolute value for the gradient; checkpoints/norm_stats.json) and is not
clipped. The same scales must be used for any new event.

Example
  python preprocess.py rain --stats ../checkpoints/norm_stats.json \
      --csv ../data/rainfall/test_2022.csv ../data/rainfall/test_2023.csv \
      ../data/rainfall/test_2024.csv --out-dir ../work/rain
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

RAIN_WINDOWS = (6, 36, 144)     # moving-sum windows in 10 min steps (1 h, 6 h, 24 h)


def read_rain_csv(path):
    """Rainfall depth per 10 min step (mm): 'Rainfall' column, else column 2."""
    df = pd.read_csv(path)
    col = df['Rainfall'] if 'Rainfall' in df.columns else df.iloc[:, 1]
    return col.to_numpy(dtype=np.float32)


def moving_sum(x, window):
    """Trailing moving sum; the first steps sum over what is available."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    i = np.arange(len(x))
    return (c[i + 1] - c[np.maximum(0, i - window + 1)]).astype(np.float32)


def rain_features(rain, stats):
    """Rainfall (T,) in mm -> normalised features (T, 6)."""
    p = np.maximum(np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    inst = np.log1p(p)
    feats = [inst] + [np.log1p(moving_sum(p, w)) for w in RAIN_WINDOWS]
    feats.append(np.log1p(np.cumsum(p)))
    feats.append(np.gradient(inst))
    raw = np.stack(feats, axis=1).astype(np.float32)
    return raw / np.asarray(stats['scales'], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('rain')
    s.add_argument('--stats', required=True, help='norm_stats.json')
    s.add_argument('--csv', required=True, nargs='+')
    s.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    with open(a.stats) as f:
        stats = json.load(f)
    if tuple(stats['rain_windows_steps']) != RAIN_WINDOWS:
        raise SystemExit('norm_stats.json does not match RAIN_WINDOWS')
    os.makedirs(a.out_dir, exist_ok=True)
    for p in a.csv:
        feat = rain_features(read_rain_csv(p), stats)
        out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + '.npy')
        np.save(out, feat)
        print(f'{os.path.basename(p)}: {feat.shape} -> {out}')


if __name__ == '__main__':
    main()
