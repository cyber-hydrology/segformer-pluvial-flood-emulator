#!/usr/bin/env python3
"""
Predict flood-depth sequences for rainfall events with a trained checkpoint.

Patch models (SegFormer, U-Net)
  The padded catchment is covered by 128 x 128 patches with a stride of 96
  cells. The static encoder runs once per patch and event; the decoder runs
  once per 10 min step with the preceding 12 steps of rainfall features.
  Flood probability and depth are recombined with Hann-weighted overlap-add,
  and depth is kept where the blended probability exceeds mask_thr (0.3).
1D-CNN
  One forward pass per step gives depth at every valid cell.
All models: depths below depth_thr (0.02 m) and cells outside valid_mask are 0.

Example
  python predict.py --checkpoint ../checkpoints/proposed_segformer_seed42.pth \
      --static ../data/static/static_inputs.npz --rain ../work/rain/test_2022.npy ../work/rain/test_2023.npy \
      ../work/rain/test_2024.npy --out-dir ../outputs/proposed
"""

import argparse
import contextlib
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_model'))
from models import build_model, cells_to_map  # noqa: E402

PAD_MULTIPLE = 32


# ----------------------------------------------------------------------------
PRECISION = 'bf16'


def autocast(device):
    """
    Mixed precision as used in training (bf16; fp16 on GPUs without bf16).
    The released models were trained and evaluated in bf16; full fp32
    inference changes the results noticeably (see README), so bf16 is the
    default on both GPU and CPU. Set PRECISION = 'fp32' to disable.
    """
    if PRECISION == 'fp32':
        return contextlib.nullcontext()
    if device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    return torch.autocast('cpu', dtype=torch.bfloat16)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(ckpt['arch'], **ckpt['model_kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval(), ckpt


def load_static(path):
    """static_inputs.npz -> static (2, H, W) float32, valid_mask (H, W) bool."""
    with np.load(path) as z:
        return z['static'].astype(np.float32), z['valid_mask'].astype(bool)


def rain_window(rain, t, seq_len):
    """Rainfall features of steps t-seq_len+1 ... t, zero-padded at the start."""
    s = t - seq_len + 1
    if s >= 0:
        return rain[s:t + 1]
    return np.concatenate([np.zeros((-s, rain.shape[1]), np.float32), rain[:t + 1]])


def pad_static(static):
    _, H, W = static.shape
    ph = -H % PAD_MULTIPLE
    pw = -W % PAD_MULTIPLE
    return np.pad(static, ((0, 0), (0, ph), (0, pw))).astype(np.float32)


def patch_origins(H, W, ps, stride):
    ys = list(range(0, H - ps + 1, stride))
    xs = list(range(0, W - ps + 1, stride))
    if ys[-1] != H - ps:
        ys.append(H - ps)
    if xs[-1] != W - ps:
        xs.append(W - ps)
    return [(y, x) for y in ys for x in xs]


# ----------------------------------------------------------------------------
class PatchPredictor:
    """Caches static-encoder features of all patches for one catchment."""

    def __init__(self, model, static, device, patch=128, stride=96, batch=16):
        self.model, self.device, self.ps, self.batch = model, device, patch, batch
        sp = torch.from_numpy(pad_static(static))
        self.H, self.W = sp.shape[1:]
        self.origins = patch_origins(self.H, self.W, patch, stride)

        chunks = []
        with torch.no_grad():
            for i in range(0, len(self.origins), batch):
                x = torch.stack([sp[:, y:y + patch, x:x + patch]
                                 for y, x in self.origins[i:i + batch]]).to(device)
                with autocast(device):
                    chunks.append([f.half() for f in model.encode_static(x)])
        self.features = [torch.cat(level) for level in zip(*chunks)]

        w = torch.hann_window(patch, periodic=False).clamp_min(0.05)
        self.blend = w[:, None] * w[None, :]

    def __call__(self, rain_win, mask_thr):
        ps = self.ps
        prob_sum = torch.zeros(self.H, self.W)
        depth_sum = torch.zeros(self.H, self.W)
        weight = torch.zeros(self.H, self.W)
        rain = torch.from_numpy(np.ascontiguousarray(rain_win, np.float32))[None].to(self.device)
        n = len(self.origins)
        with torch.no_grad():
            for i in range(0, n, self.batch):
                j = min(i + self.batch, n)
                feats = [f[i:j].float() for f in self.features]
                with autocast(self.device):
                    prob, depth = self.model.decode_with_rain(feats, rain.expand(j - i, -1, -1))
                prob, depth = prob.float().cpu()[:, 0], depth.float().cpu()[:, 0]
                for k, (y, x) in enumerate(self.origins[i:j]):
                    prob_sum[y:y + ps, x:x + ps] += prob[k] * self.blend
                    depth_sum[y:y + ps, x:x + ps] += depth[k] * self.blend
                    weight[y:y + ps, x:x + ps] += self.blend
        weight = weight.clamp_min(1e-6)
        out = ((prob_sum / weight) > mask_thr).float() * (depth_sum / weight)
        return out.numpy()


def predict_event(model, ckpt, static, valid, rain, device):
    """Returns predicted depth (T, H, W) in metres and the run time in seconds."""
    cfg = ckpt['config']
    H, W = valid.shape
    T = len(rain)
    pred = np.zeros((T, H, W), np.float32)
    t0 = time.time()

    if ckpt['arch'] == 'cnn1d':
        idx = torch.from_numpy(np.flatnonzero(valid.ravel())).long().to(device)
        with torch.no_grad():
            for t in range(T):
                r = torch.from_numpy(rain_window(rain, t, cfg['seq_len']))[None].to(device)
                with autocast(device):
                    d = cells_to_map(model(r), idx, H, W)
                pred[t] = d.float().cpu().numpy()[0, 0]
    else:
        predictor = PatchPredictor(model, static, device, cfg['patch_size'], cfg['patch_stride'])
        for t in range(T):
            pred[t] = predictor(rain_window(rain, t, cfg['seq_len']), cfg['mask_thr'])[:H, :W]

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    pred[pred < cfg['depth_thr']] = 0.0
    pred[:, ~valid] = 0.0
    return pred, elapsed


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--static', required=True, help='static_inputs.npz')
    ap.add_argument('--rain', required=True, nargs='+', help='rainfall feature .npy file(s)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    a = ap.parse_args()

    global PRECISION
    PRECISION = a.precision

    device = torch.device(a.device)
    model, ckpt = load_checkpoint(a.checkpoint, device)
    static, valid = load_static(a.static)
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"model {ckpt['arch']}  device {device}")

    for path in a.rain:
        rain = np.load(path).astype(np.float32)
        pred, sec = predict_event(model, ckpt, static, valid, rain, device)
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(a.out_dir, f'{name}.npz')
        np.savez_compressed(out, depth=pred, valid_mask=valid)
        print(f'{name}: {len(rain)} steps in {sec:.1f} s -> {out}')


if __name__ == '__main__':
    main()
