"""
DoA estimation for mixture.wav using SRP-PHAT on the raw 4-channel signal.

Uses all 6 microphone pairs to build a power map over azimuths 0–360°,
then finds the N strongest peaks (one per speaker) with a minimum angular
separation so we don't double-count the same speaker.

No separation required — runs directly on the mixture.

Usage:
    python analysis/doa/estimate_doa_mixture.py [--n_src 7] [--plot]
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
parser.add_argument("--n_src",   type=int, default=7,   help="Expected number of speakers")
parser.add_argument("--min_sep", type=int, default=20,  help="Min angular separation between peaks (degrees)")
parser.add_argument("--plot",    action="store_true",   help="Show polar plot")
args = parser.parse_args()

# ── Load geometry ─────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

# 4-mic layout: [left-front, left-rear, right-front, right-rear]
MIC_POS = np.array([
    [ D/2,  L/2],   # left-front
    [ D/2, -L/2],   # left-rear
    [-D/2,  L/2],   # right-front
    [-D/2, -L/2],   # right-rear
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))


def gcc_phat(x: np.ndarray, y: np.ndarray, n_fft: int) -> np.ndarray:
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa(ch_a: int, ch_b: int, phi_rad: float) -> float:
    """Expected time delay (samples) for a plane wave arriving from azimuth phi."""
    dx = MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]
    dy = MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]
    return (dx * np.sin(phi_rad) + dy * np.cos(phi_rad)) / C


def srp_phat_power(channels: list, sr: int,
                   azimuths: np.ndarray) -> np.ndarray:
    """Compute SRP-PHAT power at each azimuth using all mic pairs."""
    n = min(len(ch) for ch in channels)
    channels = [ch[:n] for ch in channels]
    n_fft = 1 << (n - 1).bit_length()

    gcc_store = {
        (a, b): gcc_phat(channels[a], channels[b], n_fft)
        for a, b in ALL_PAIRS
    }
    power = np.zeros(len(azimuths))
    for idx, az in enumerate(azimuths):
        phi = np.deg2rad(az)
        for a, b in ALL_PAIRS:
            tau = predicted_tdoa(a, b, phi)
            lag = tau * sr + n_fft // 2
            lo  = int(np.clip(np.floor(lag), 0, n_fft - 1))
            hi  = int(np.clip(lo + 1,        0, n_fft - 1))
            frac = lag - np.floor(lag)
            power[idx] += ((1 - frac) * gcc_store[(a, b)][lo]
                           + frac     * gcc_store[(a, b)][hi])
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

print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | {data.shape[0]/sr:.1f}s")

channels = [data[:, k] for k in range(data.shape[1])]

# ── SRP-PHAT power map ────────────────────────────────────────────────────────
AZIMUTHS = np.linspace(0, 360, 720, endpoint=False)   # 0.5° resolution
print(f"Computing SRP-PHAT power map ({len(AZIMUTHS)} azimuths) ...")
power = srp_phat_power(channels, sr, AZIMUTHS)

# Normalise
power -= power.min()
power /= power.max() + 1e-12

# ── Peak detection ────────────────────────────────────────────────────────────
# Minimum separation in samples of the azimuth grid
min_sep_samples = int(args.min_sep / (AZIMUTHS[1] - AZIMUTHS[0]))
peaks, props = find_peaks(power, distance=min_sep_samples, height=0.1)
# Sort by height, keep top N
order = np.argsort(props["peak_heights"])[::-1]
top_peaks = peaks[order[:args.n_src]]
top_azimuths = AZIMUTHS[top_peaks]
top_powers   = props["peak_heights"][order[:args.n_src]]

# Sort by azimuth for readability
sort_idx     = np.argsort(top_azimuths)
top_azimuths = top_azimuths[sort_idx]
top_powers   = top_powers[sort_idx]

print(f"\n=== SRP-PHAT DoA estimates (n_src={args.n_src}) ===")
for i, (az, pw) in enumerate(zip(top_azimuths, top_powers)):
    print(f"  Speaker {i+1}: {az:6.1f}°  (relative power {pw:.3f})")

# ── Save results ──────────────────────────────────────────────────────────────
result = {
    "n_src": args.n_src,
    "method": "SRP-PHAT on raw mixture",
    "speakers": [
        {"speaker": i + 1, "azimuth_deg": float(f"{az:.1f}"), "relative_power": float(f"{pw:.4f}")}
        for i, (az, pw) in enumerate(zip(top_azimuths, top_powers))
    ]
}
out_json = os.path.join(OUT_DIR, "doa_estimates.json")
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {os.path.relpath(out_json)}")

# ── Optional plot ─────────────────────────────────────────────────────────────
if args.plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    phi_rad = np.deg2rad(AZIMUTHS)
    ax.plot(phi_rad, power, color="steelblue", linewidth=0.8, alpha=0.7)
    for i, (az, pw) in enumerate(zip(top_azimuths, top_powers)):
        ax.annotate(f"S{i+1}\n{az:.0f}°",
                    xy=(np.deg2rad(az), pw),
                    fontsize=9, ha="center",
                    xytext=(np.deg2rad(az), pw + 0.08))
        ax.plot(np.deg2rad(az), pw, "ro", markersize=8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)   # CCW: 0°=front, 90°=left, 270°=right
    ax.set_title(f"SRP-PHAT power map — mixture.wav\n(top {args.n_src} peaks)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "doa_polar.png"), dpi=150)
    print(f"Plot saved → analysis/doa/doa_polar.png")
    plt.show()
