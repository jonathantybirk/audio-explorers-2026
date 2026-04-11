"""
Time-windowed SRP-PHAT on the raw 4-channel mixture.

Divides the clip into overlapping windows and computes a SRP-PHAT DoA
estimate per window. Saves a heatmap image and a JSON with per-window data.

Use this to map transcript-timed utterances to DoA:
  - identify roughly when speaker X is talking (in seconds)
  - read off the dominant azimuth from that time window

Usage:
    python analysis/doa/windowed_srp_doa.py [--win_s 1.0] [--hop_s 0.5] [--plot]
"""

import argparse
import itertools
import json
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR   = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--win_s",  type=float, default=1.0,  help="Window length in seconds")
parser.add_argument("--hop_s",  type=float, default=0.25, help="Hop size in seconds")
parser.add_argument("--az_res", type=float, default=1.0,  help="Azimuth resolution in degrees")
parser.add_argument("--plot",   action="store_true")
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


def predicted_tdoa(ch_a, ch_b, phi_rad):
    dx = MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]
    dy = MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]
    return (dx * np.sin(phi_rad) + dy * np.cos(phi_rad)) / C


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def srp_power_window(channels_win, sr):
    n = min(len(c) for c in channels_win)
    channels_win = [c[:n] for c in channels_win]
    n_fft = 1 << (n - 1).bit_length()

    gcc_store = {
        (a, b): gcc_phat(channels_win[a], channels_win[b], n_fft)
        for a, b in ALL_PAIRS
    }
    power = np.zeros(len(AZIMUTHS))
    for idx, az in enumerate(AZIMUTHS):
        phi = np.deg2rad(az)
        for a, b in ALL_PAIRS:
            tau   = predicted_tdoa(a, b, phi)
            lag   = tau * sr + n_fft // 2
            lo    = int(np.clip(np.floor(lag), 0, n_fft - 1))
            hi    = int(np.clip(lo + 1, 0, n_fft - 1))
            frac  = lag - np.floor(lag)
            power[idx] += (1 - frac) * gcc_store[(a, b)][lo] + frac * gcc_store[(a, b)][hi]
    return power


# ── Load audio ────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = wavfile.read(WAV_PATH)
if data.dtype == np.int16:
    data = data.astype(np.float64) / 32768.0
elif data.dtype == np.int32:
    data = data.astype(np.float64) / 2**31
else:
    data = data.astype(np.float64)

n_total = data.shape[0]
duration = n_total / sr
print(f"  {n_total} samples | {data.shape[1]} ch | {sr} Hz | {duration:.1f}s")

channels = [data[:, k] for k in range(4)]

win_n = int(args.win_s * sr)
hop_n = int(args.hop_s * sr)

# ── Windowed SRP-PHAT ─────────────────────────────────────────────────────────
starts = list(range(0, n_total - win_n, hop_n))
print(f"Computing SRP-PHAT over {len(starts)} windows "
      f"({args.win_s}s win, {args.hop_s}s hop) ...")

time_centres = []
peak_azimuths = []
srp_matrix = []  # (n_windows, n_az)

for i, start in enumerate(starts):
    end  = start + win_n
    win  = [ch[start:end] for ch in channels]
    pwr  = srp_power_window(win, sr)
    pwr -= pwr.min()
    pwr /= pwr.max() + 1e-12

    t_c = (start + win_n / 2) / sr
    peaks_idx, _ = find_peaks(pwr, distance=int(15 / args.az_res), height=0.2)
    top = peaks_idx[np.argsort(pwr[peaks_idx])[::-1][:3]] if len(peaks_idx) else [np.argmax(pwr)]
    top_az = AZIMUTHS[top]

    time_centres.append(round(t_c, 3))
    peak_azimuths.append([float(az) for az in top_az])
    srp_matrix.append(pwr.tolist())

    if (i + 1) % 10 == 0 or i == len(starts) - 1:
        print(f"  {i+1}/{len(starts)}  t={t_c:.1f}s  peaks={list(top_az.round(0))}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_json = os.path.join(OUT_DIR, "windowed_doa.json")
with open(out_json, "w") as f:
    json.dump({
        "win_s": args.win_s,
        "hop_s": args.hop_s,
        "az_res": args.az_res,
        "azimuths": AZIMUTHS.tolist(),
        "windows": [
            {"t_centre": t, "top_peaks_deg": p, "srp_power": s}
            for t, p, s in zip(time_centres, peak_azimuths, srp_matrix)
        ]
    }, f)
print(f"\nSaved → {os.path.relpath(out_json)}")

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'t (s)':>7}  top-3 DoA peaks")
print("-" * 45)
for t, p in zip(time_centres, peak_azimuths):
    pstr = " / ".join(f"{az:.0f}°" for az in p)
    print(f"{t:>7.2f}  {pstr}")

# ── Optional heatmap ──────────────────────────────────────────────────────────
if args.plot:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mat = np.array(srp_matrix).T   # (n_az, n_windows)

    fig, ax = plt.subplots(figsize=(14, 6))
    extent = [time_centres[0], time_centres[-1], AZIMUTHS[-1], AZIMUTHS[0]]
    im = ax.imshow(mat, aspect="auto", extent=extent,
                   cmap="inferno", norm=mcolors.PowerNorm(gamma=0.5))
    plt.colorbar(im, ax=ax, label="Normalised SRP power")

    # Mark known SRP-PHAT peaks from full-mixture analysis
    known_peaks = [2, 25, 90, 178, 270, 332.5, 358]
    for az in known_peaks:
        ax.axhline(az, color="cyan", linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Azimuth (°)  [0=front, 90=left, 270=right]")
    ax.set_title("Time-windowed SRP-PHAT — mixture.wav")
    ax.yaxis.set_ticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.yaxis.set_ticklabels(["0° front", "45°", "90° left", "135°", "180° back",
                              "225°", "270° right", "315°", "360°"])
    plt.tight_layout()
    out_img = os.path.join(OUT_DIR, "windowed_doa_heatmap.png")
    plt.savefig(out_img, dpi=150)
    print(f"Heatmap saved → {os.path.relpath(out_img)}")
    plt.show()
