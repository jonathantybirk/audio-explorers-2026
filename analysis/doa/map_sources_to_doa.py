"""
Map each separated mono source to a DoA angle.

Strategy: for each separated source s(t), compute GCC-PHAT between s(t)
and each of the 4 mixture channels. The peak lag gives the delay of the
source at that channel. We then build a predicted-vs-measured TDOA match
across all 6 channel pairs to find the best-fitting azimuth.

Usage:
    python analysis/doa/map_sources_to_doa.py \
        [--src_dir analysis/ica/separated] \
        [--pattern "mixture_fmnmf2_opt_n7_source_*.wav"]
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy.io import wavfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", default=os.path.join(REPO_ROOT, "analysis", "ica", "separated"))
parser.add_argument("--pattern", default="mixture_fmnmf2_opt_n7_source_*.wav")
args = parser.parse_args()

# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [ D/2,  L/2],   # LF
    [ D/2, -L/2],   # LR
    [-D/2,  L/2],   # RF
    [-D/2, -L/2],   # RR
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))
AZIMUTHS  = np.linspace(0, 360, 720, endpoint=False)


def gcc_phat_cross(x: np.ndarray, y: np.ndarray, n_fft: int) -> np.ndarray:
    """GCC-PHAT between two signals. Returns full cross-correlation (fftshifted)."""
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa_samples(ch_a: int, ch_b: int, phi_rad: float, sr: int) -> float:
    dx = MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]
    dy = MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]
    tau_s = (dx * np.sin(phi_rad) + dy * np.cos(phi_rad)) / C
    return tau_s * sr


def srp_from_source(src: np.ndarray, mix_chs: list, sr: int) -> tuple:
    """
    For each pair (a, b), compute GCC-PHAT(src, mix_ch_a) and GCC-PHAT(src, mix_ch_b).
    The measured TDOA between ch_a and ch_b (as seen by the source) is the difference
    of the two peak lags. We then scan azimuths and pick the one whose predicted TDOAs
    best match the measured ones.
    """
    n = min(len(src), min(len(ch) for ch in mix_chs))
    src    = src[:n]
    mix_chs = [ch[:n] for ch in mix_chs]
    n_fft  = 1 << (n - 1).bit_length()
    center = n_fft // 2

    # GCC-PHAT between source and each individual mixture channel
    gcc_src_ch = [gcc_phat_cross(src, mix_chs[k], n_fft) for k in range(4)]

    # Measured lag of source relative to each channel (peak position relative to center)
    measured_lag = np.array([
        np.argmax(gcc_src_ch[k]) - center for k in range(4)
    ], dtype=float)

    # For each pair (a,b): measured TDOA = lag_a - lag_b
    measured_tdoa = {(a, b): measured_lag[a] - measured_lag[b] for a, b in ALL_PAIRS}

    # SRP: scan azimuths, score = negative sum of squared residuals between
    # predicted and measured TDOAs across all pairs
    scores = np.zeros(len(AZIMUTHS))
    for idx, az in enumerate(AZIMUTHS):
        phi = np.deg2rad(az)
        residual = 0.0
        for a, b in ALL_PAIRS:
            pred = predicted_tdoa_samples(a, b, phi, sr)
            residual += (pred - measured_tdoa[(a, b)]) ** 2
        scores[idx] = -residual

    best_idx = np.argmax(scores)
    best_az  = AZIMUTHS[best_idx]

    # Also return the raw power-map-style score normalised for display
    scores -= scores.min()
    scores /= scores.max() + 1e-12

    return best_az, scores


# ── Load mixture channels ─────────────────────────────────────────────────────
print("Loading mixture.wav ...")
sr_mix, mix_data = wavfile.read(WAV_PATH)
if mix_data.dtype == np.int16:
    mix_data = mix_data.astype(np.float64) / 32768.0
elif mix_data.dtype == np.int32:
    mix_data = mix_data.astype(np.float64) / 2**31
else:
    mix_data = mix_data.astype(np.float64)
mix_channels = [mix_data[:, k] for k in range(4)]

# ── Process each source file ──────────────────────────────────────────────────
pattern = os.path.join(args.src_dir, args.pattern)
files   = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No files matched: {pattern}")

print(f"\nFound {len(files)} source files.\n")
print(f"{'File':<50} {'Est. DoA':>9}")
print("-" * 61)

results = []
for fpath in files:
    fname = os.path.basename(fpath)
    _, src_data = wavfile.read(fpath)
    if src_data.dtype == np.int16:
        src = src_data.astype(np.float64) / 32768.0
    elif src_data.dtype == np.int32:
        src = src_data.astype(np.float64) / 2**31
    else:
        src = src_data.astype(np.float64)

    if src.ndim > 1:
        src = src.mean(axis=1)

    best_az, _ = srp_from_source(src, mix_channels, sr_mix)
    results.append({"file": fname, "azimuth_deg": float(best_az)})
    print(f"{fname:<50} {best_az:>8.1f}°")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "source_doa_map.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {os.path.relpath(out_path)}")
