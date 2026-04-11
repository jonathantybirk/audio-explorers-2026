"""
Masked SRP-PHAT: estimate DoA for each separated source using soft T-F masks.

For each separated source k:
  1. Compute a Wiener soft mask from all source spectrograms:
       M_k(t,f) = |S_k(t,f)|^2 / (sum_j |S_j(t,f)|^2 + eps)
  2. Apply the mask to each of the 4 mixture channels (preserving spatial cues)
  3. Run SRP-PHAT on the 4 masked channels to estimate the source DoA

This works because the mask selects time-frequency bins where source k dominates,
and the mixture channels at those bins carry the inter-mic delays for that source.

Usage:
    python analysis/doa/masked_srp_doa.py \
        [--pattern "mixture_fmnmf2_opt_n7_source_*.wav"] \
        [--n_fft 2048] [--hop 512]
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy.io import wavfile
import scipy.signal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
SRC_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
OUT_DIR   = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default="mixture_fmnmf2_opt_n7_source_*.wav")
parser.add_argument("--n_fft",   type=int, default=2048)
parser.add_argument("--hop",     type=int, default=512)
parser.add_argument("--az_res",  type=float, default=0.5, help="Azimuth grid resolution in degrees")
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
AZIMUTHS  = np.arange(0, 360, args.az_res)


def predicted_tdoa_samples(ch_a, ch_b, phi_rad, sr):
    dx = MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]
    dy = MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]
    return (dx * np.sin(phi_rad) + dy * np.cos(phi_rad)) / C * sr


# ── Load mixture ──────────────────────────────────────────────────────────────
print("Loading mixture.wav ...")
sr, mix_data = wavfile.read(WAV_PATH)
if mix_data.dtype == np.int16:
    mix_data = mix_data.astype(np.float64) / 32768.0
elif mix_data.dtype == np.int32:
    mix_data = mix_data.astype(np.float64) / 2**31
else:
    mix_data = mix_data.astype(np.float64)

n_samples = mix_data.shape[0]
mix_chs = [mix_data[:, k] for k in range(4)]

# STFTs of mixture channels
window = scipy.signal.get_window("hann", args.n_fft)
mix_stft = [
    scipy.signal.stft(ch, fs=sr, window=window, nperseg=args.n_fft,
                      noverlap=args.n_fft - args.hop, boundary=None)[2]
    for ch in mix_chs
]  # each: (n_freq, n_frames) complex

# ── Load separated sources ────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(SRC_DIR, args.pattern)))
if not files:
    raise FileNotFoundError(f"No files found: {os.path.join(SRC_DIR, args.pattern)}")
print(f"Found {len(files)} source files.")

src_stfts = []
fnames = []
for fpath in files:
    fnames.append(os.path.basename(fpath))
    _, data = wavfile.read(fpath)
    if data.dtype == np.int16:
        sig = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        sig = data.astype(np.float64) / 2**31
    else:
        sig = data.astype(np.float64)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    # Trim/pad to match mixture length
    if len(sig) > n_samples:
        sig = sig[:n_samples]
    else:
        sig = np.pad(sig, (0, n_samples - len(sig)))
    _, _, S = scipy.signal.stft(sig, fs=sr, window=window, nperseg=args.n_fft,
                                 noverlap=args.n_fft - args.hop, boundary=None)
    src_stfts.append(S)  # (n_freq, n_frames) complex

n_src = len(src_stfts)

# ── Soft Wiener masks ─────────────────────────────────────────────────────────
# M_k(t,f) = |S_k|^2 / (sum_j |S_j|^2 + eps)
src_power = np.stack([np.abs(S)**2 for S in src_stfts], axis=0)  # (K, n_freq, n_frames)
total_power = src_power.sum(axis=0) + 1e-12  # (n_freq, n_frames)
masks = src_power / total_power  # (K, n_freq, n_frames)

# ── Masked SRP-PHAT for each source ──────────────────────────────────────────
print(f"\n{'Source file':<50} {'Est. DoA':>9}  (top-3)")
print("-" * 72)

results = []
for k, fname in enumerate(fnames):
    mask_k = masks[k]  # (n_freq, n_frames)

    # Apply mask to each mixture channel STFT
    masked_mix = [mix_stft[ch] * mask_k for ch in range(4)]  # list of (n_freq, n_frames) complex

    # GCC-PHAT in frequency domain for each pair using the masked channels
    # GCC(a,b)(f) = X_a(f) * conj(X_b(f)) / |X_a(f) * conj(X_b(f))| per frame
    # Then SRP: sum over frequency bins weighted by the mask

    # Build SRP power over azimuths
    srp_power = np.zeros(len(AZIMUTHS))

    for idx, az in enumerate(AZIMUTHS):
        phi = np.deg2rad(az)
        for a, b in ALL_PAIRS:
            tau = predicted_tdoa_samples(a, b, phi, sr)
            # Phase-shift steering: e^{-j 2pi f tau / N}
            freq_bins = np.arange(args.n_fft // 2 + 1)
            steering = np.exp(-1j * 2 * np.pi * freq_bins * tau / args.n_fft)  # (n_freq,)
            # Cross-spectral density between masked channels, summed over frames
            Xa = masked_mix[a]   # (n_freq, n_frames)
            Xb = masked_mix[b]   # (n_freq, n_frames)
            cross = Xa * np.conj(Xb)                         # (n_freq, n_frames)
            phat  = cross / (np.abs(cross) + 1e-12)          # normalise
            # Steer and sum over freq and frames
            srp_power[idx] += np.real((steering[:, None] * phat).sum())

    # Find peak
    best_idx = np.argmax(srp_power)
    best_az  = AZIMUTHS[best_idx]

    # Top-3 peaks
    top3 = np.argsort(srp_power)[::-1][:3]
    top3_str = " / ".join(f"{AZIMUTHS[i]:.0f}°" for i in top3)

    results.append({"file": fname, "azimuth_deg": float(best_az)})
    print(f"{fname:<50} {best_az:>8.1f}°  [{top3_str}]")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "source_doa_map.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {os.path.relpath(out_path)}")
