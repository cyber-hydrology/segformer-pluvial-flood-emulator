#!/usr/bin/env python3
"""
Event metrics against the reference simulation.

All metrics use the valid cells over all 10 min steps of an event (pooled
contingency table and pooled errors). Extent metrics use a 0.10 m threshold.

  CSI, POD, FAR, HSS     extent agreement
  RMSE_domain_cm         depth RMSE over all valid cells
  RMSE_wet_cm            depth RMSE over cells with reference depth >= 0.10 m
  bias_cm                mean(pred - ref)
  vol_ratio              sum(pred) / sum(ref), no threshold
  area_ratio             (TP + FP) / (TP + FN)
  BND_mean_m             mean distance from the predicted to the reference
                         inundation boundary (steps with >= 50 wet cells)
  BND_gt1_frac           share of predicted boundary cells displaced > 1 cell
  FSS_30m                fractions skill score in a 3 x 3 cell window

Example
  python evaluate.py --pred ../outputs/proposed/test_2022.npz \
      ../outputs/proposed/test_2023.npz ../outputs/proposed/test_2024.npz \
      --ref ../data/reference/H12_2022.npz ../data/reference/H12_2023.npz \
      ../data/reference/H12_2024.npz --out ../outputs/proposed/metrics.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, uniform_filter

THR = 0.10          # m
CELL = 10.0         # m
MIN_WET = 50        # cells; steps used for boundary and FSS
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def contingency(obs, prd):
    tp = int((obs & prd).sum())
    fp = int((~obs & prd).sum())
    fn = int((obs & ~prd).sum())
    tn = int(obs.size - tp - fp - fn)
    n = obs.size
    expected = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    return {'CSI': tp / (tp + fp + fn), 'POD': tp / (tp + fn), 'FAR': fp / (tp + fp),
            'HSS': (tp + tn - expected) / (n - expected), 'area_ratio': (tp + fp) / (tp + fn)}


def boundary(mask):
    return mask & ~binary_erosion(mask, structure=CROSS, border_value=0)


def boundary_distances(obs, prd, valid):
    o, p = boundary(obs & valid), boundary(prd & valid)
    if not o.any() or not p.any():
        return np.empty(0)
    return distance_transform_edt(~o, sampling=CELL)[p]


def event_metrics(pred, ref, valid):
    P, R = pred[:, valid], ref[:, valid]
    m = contingency(R >= THR, P >= THR)
    d = (P - R) * 100.0
    wet = R >= THR
    m.update(RMSE_domain_cm=float(np.sqrt((d ** 2).mean())),
             RMSE_wet_cm=float(np.sqrt((d[wet] ** 2).mean())),
             bias_cm=float(d.mean()),
             vol_ratio=float(P.sum() / R.sum()))

    steps = np.flatnonzero(wet.sum(1) >= MIN_WET)
    dist, num, den = [], 0.0, 0.0
    for t in steps:
        ob, pb = ref[t] >= THR, pred[t] >= THR
        fo = uniform_filter(ob.astype(np.float32), 3, mode='constant')
        fp = uniform_filter(pb.astype(np.float32), 3, mode='constant')
        num += float(((fp - fo) ** 2)[valid].sum())
        den += float((fp ** 2 + fo ** 2)[valid].sum())
        if (pb & valid).sum() >= MIN_WET:
            dist.append(boundary_distances(ob, pb, valid))
    dist = np.concatenate(dist) if dist else np.empty(0)
    m.update(BND_mean_m=float(dist.mean()) if dist.size else np.nan,
             BND_gt1_frac=float((dist > CELL).mean()) if dist.size else np.nan,
             FSS_30m=1 - num / den if den > 0 else np.nan)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, nargs='+', help='prediction .npz (key "depth")')
    ap.add_argument('--ref', required=True, nargs='+', help='reference .npz (key "depth"), same order')
    ap.add_argument('--out', default=None, help='CSV file')
    a = ap.parse_args()
    if len(a.pred) != len(a.ref):
        raise SystemExit('--pred and --ref need the same number of files')

    rows = []
    for pp, rp in zip(a.pred, a.ref):
        p, r = np.load(pp), np.load(rp)
        valid = r['valid_mask'].astype(bool)
        n = min(len(p['depth']), len(r['depth']))
        m = event_metrics(p['depth'][:n].astype(np.float32), r['depth'][:n].astype(np.float32), valid)
        rows.append({'event': os.path.splitext(os.path.basename(rp))[0], **m})

    df = pd.DataFrame(rows)
    df = pd.concat([df, df.mean(numeric_only=True).to_frame().T.assign(event='mean')],
                   ignore_index=True)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
