# SegFormer emulator for dynamic urban pluvial flooding

Code of a deep-learning emulator for urban pluvial flood depth. The emulator predicts a 10 m flood-depth map every 10 min from a 2 h history of rainfall features and two static maps (DEM, land use). It was trained on 90 synthetic storms simulated with the H12 hydrodynamic model for Gunsan, Republic of Korea, and tested on three historical storms (2022, 2023, 2024). With the trained weights and test data (see below), this repository reproduces the test results of the proposed model and the two benchmarks (U-Net, 1D-CNN).

## Contents

```
01_data_preprocessing/
    preprocess.py        rainfall CSV -> normalised rainfall features
02_model/
    models.py            FloodSegFormer (proposed), FloodUNet, RainfallCNN1D
03_inference/
    predict.py           full-map prediction for rainfall events
    evaluate.py          event metrics against the reference simulation
checkpoints/             (in the release package, see below)
    proposed_segformer_seed42.pth
    unet_seed42.pth
    cnn1d_seed42.pth
    norm_stats.json      rainfall feature scales from the 90 training storms
data/                    (in the release package, see below)
    static/              static_inputs.npz (normalised DEM and land use, valid mask)
    rainfall/            test_2022.csv, test_2023.csv, test_2024.csv
    reference/           H12_2022.npz, H12_2023.npz, H12_2024.npz
reproduce.sh             runs all steps below
```

## Trained weights and test data

The trained weights (`checkpoints/`) and test data (`data/`) are too large for the repository and are distributed as one package (`Emulator_Floodsegformer.zip`, 355 MB) that contains the code, weights and data.

Download it from the GitHub release v1.0.0: [Emulator_Floodsegformer.zip](https://github.com/cyber-hydrology/segformer-pluvial-flood-emulator/releases/download/v1.0.0/Emulator_Floodsegformer.zip)

Unzip the package and run everything inside the extracted folder:

```
unzip Emulator_Floodsegformer.zip
cd urban_flood_emulator_v1.0.0
bash reproduce.sh
```

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

| | Training of the released weights | Inference check of this archive |
|---|---|---|
| Hardware | one NVIDIA H100 NVL GPU | laptop CPU |
| Software | PyTorch 2.7.1 with CUDA 11.8 | Python 3.10.11, PyTorch 2.14.1 (CPU), NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 |
| Precision | bf16 mixed precision | bf16 (default of `predict.py`) |

A GPU is optional; for GPU use, install the CUDA build of PyTorch for your system before `requirements.txt`.

## Reproducing the test results

```
# 1) rainfall features (static inputs are provided ready to use)
python 01_data_preprocessing/preprocess.py rain --stats checkpoints/norm_stats.json \
    --csv data/rainfall/test_2022.csv data/rainfall/test_2023.csv data/rainfall/test_2024.csv \
    --out-dir work/rain

# 2) prediction (repeat with unet_seed42 and cnn1d_seed42)
python 03_inference/predict.py --checkpoint checkpoints/proposed_segformer_seed42.pth \
    --static data/static/static_inputs.npz --rain work/rain/test_2022.npy work/rain/test_2023.npy work/rain/test_2024.npy \
    --out-dir outputs/proposed_segformer_seed42

# 3) metrics
python 03_inference/evaluate.py \
    --pred outputs/proposed_segformer_seed42/test_2022.npz outputs/proposed_segformer_seed42/test_2023.npz outputs/proposed_segformer_seed42/test_2024.npz \
    --ref data/reference/H12_2022.npz data/reference/H12_2023.npz data/reference/H12_2024.npz
```

or simply `bash reproduce.sh`.

Reference results, mean over the three events (computed on a GPU with the released weights):

| Model | CSI | POD | FAR | HSS | RMSE (cm) | RMSE wet (cm) | Volume ratio | Area ratio | Boundary displacement (m) | FSS 30 m |
|---|---|---|---|---|---|---|---|---|---|---|
| Proposed | 0.947 | 0.966 | 0.020 | 0.970 | 0.73 | 1.85 | 0.967 | 0.986 | 0.65 | 0.992 |
| U-Net | 0.913 | 0.949 | 0.039 | 0.950 | 0.96 | 2.49 | 0.963 | 0.988 | 1.13 | 0.987 |
| 1D-CNN | 0.909 | 0.958 | 0.054 | 0.947 | 1.50 | 3.65 | 0.968 | 1.012 | 2.32 | 0.983 |

`predict.py` runs in bf16 mixed precision, as used in training. Small differences from the table are expected on other hardware.

## Data

| Item | Description |
|---|---|
| `static_inputs.npz` | Model-ready static inputs on the 485 x 504 grid (10 m): `static` (2, 485, 504) float32 with channel 0 = DEM min-max normalised to 0-1 over the study area and channel 1 = land-use class code / 30, and `valid_mask` (485, 504) bool, False outside the study area and on river channels (excluded from training and evaluation) |
| `test_YYYY.csv` | 10 min rainfall depth (mm) at the Gunsan station of the Korea Meteorological Administration, 288 steps (48 h): 2022-08-10 00:10 to 08-12 00:00, 2023-07-13 00:10 to 07-15 00:00, 2024-07-09 00:10 to 07-11 00:00 |
| `H12_YYYY.npz` | H12 reference flood depth (m), keys `depth` (288, 485, 504) and `valid_mask` (485, 504) |

Rainfall features (`preprocess.py`): log1p of the 10 min rainfall, of its trailing 1 h, 6 h and 24 h sums and of the cumulative event rainfall, and the temporal gradient of log1p rainfall. Each feature is divided by its 99th percentile over the 90 training storms (`checkpoints/norm_stats.json`); new events must use the same scales.

The original DEM and land-use grids are not redistributed; only the normalised model inputs above are provided, without georeferencing. They are sufficient to run and evaluate the released models.

Other data and code are available from the authors on request.

## Models and settings

| Checkpoint | Parameters | Selection |
|---|---|---|
| `proposed_segformer_seed42.pth` | 6,963,749 | highest validation CSI (0.958) |
| `unet_seed42.pth` | 4,180,482 | highest validation CSI (0.923) |
| `cnn1d_seed42.pth` | 63,935,491 | lowest validation loss |

Inference settings (stored in each checkpoint under `config`): 12 steps (2 h) of rainfall history; for the SegFormer and U-Net, 128 x 128 patches with a stride of 96 cells, recombined with Hann-weighted overlap-add, and depth kept where the blended flood probability exceeds 0.3.

## Scope and limitations

- The models are specific to the Gunsan study area, its 10 m grid and the static inputs in `data/static`. They are not expected to work for other catchments, grids or cell sizes without retraining.
- The models emulate H12 simulations and inherit their assumptions and errors. They are not a substitute for a hydrodynamic model or for observations.
- The code and weights are provided for research and reproducibility only, without warranty, and are not intended for operational flood warning.

## License

Code: MIT License (`LICENSE`). Trained weights and data in the release package: CC BY-NC 4.0.
