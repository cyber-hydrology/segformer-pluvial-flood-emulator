#!/bin/bash
# Reproduce the test results of the three released models.
set -e
cd "$(dirname "$0")"
EVENTS="2022 2023 2024"

STATIC=data/static/static_inputs.npz   # normalised DEM, land use and valid mask

python 01_data_preprocessing/preprocess.py rain --stats checkpoints/norm_stats.json \
    --csv $(for y in $EVENTS; do echo data/rainfall/test_$y.csv; done) --out-dir work/rain

for m in proposed_segformer_seed42 unet_seed42 cnn1d_seed42; do
    echo "== $m"
    python 03_inference/predict.py --checkpoint checkpoints/$m.pth --static $STATIC \
        --rain $(for y in $EVENTS; do echo work/rain/test_$y.npy; done) --out-dir outputs/$m
    python 03_inference/evaluate.py \
        --pred $(for y in $EVENTS; do echo outputs/$m/test_$y.npz; done) \
        --ref $(for y in $EVENTS; do echo data/reference/H12_$y.npz; done) \
        --out outputs/$m/metrics.csv
done
