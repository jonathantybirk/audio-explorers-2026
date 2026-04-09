"""
Time-azimuth DoA heatmap for mixture.wav.

Splits the recording into short windows, runs SRP-PHAT on each,
and stacks the results into a (time × azimuth) heatmap.

This is raw data — no peak picking, no interpretation.
Each horizontal stripe shows the spatial energy at that moment in time.
Bright vertical bands = a speaker active at that angle throughout the clip.
"""

import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV  = os.path.join(REPO, "DONT-TOUCH/Software Case/mixture.wav")
GEO  = os.path.join(REPO, "data/mic_geometry.json")
OUT  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

WINDOW_S  = 0.3    # seconds per SRP-PHAT window
STEP_S    = 0.1    # step between windows (overlap)
AZ_STEP   = 1.0    # azimuth resolution (degrees)

with open(GEO) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [ D/2,  L/2],
    [ D/2, -L/2],
    [-D/2,  L/2],
    [-D/2, -L/2],
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    else:
        data = data.astype(np.float64)
    return sr, data


def gcc_phat_power_at_az(channels, sr, az_deg, n_fft):
    phi = np.deg2rad(az_deg)
    score = 0.0
    for ch_a, ch_b in ALL_PAIRS:
        tau = (  (MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]) * np.sin(phi)
               + (MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]) * np.cos(phi)) / C
        lag_idx = tau * sr + n_fft // 2
        lo = int(np.floor(lag_idx)); hi = lo + 1; frac = lag_idx - lo
        lo = np.clip(lo, 0, n_fft - 1); hi = np.clip(hi, 0, n_fft - 1)

        X = np.fft.rfft(channels[ch_a], n=n_fft)
        Y = np.fft.rfft(channels[ch_b], n=n_fft)
        G = X * np.conj(Y)
        G /= np.abs(G) + 1e-12
        gcc = np.fft.fftshift(np.fft.irfft(G, n=n_fft))
        score += (1 - frac) * gcc[lo] + frac * gcc[hi]
    return score


def main():
    print("Loading mixture.wav ...")
    sr, data = load_wav(WAV)
    n_total = data.shape[0]
    win_len  = int(WINDOW_S * sr)
    step_len = int(STEP_S * sr)
    azimuths = np.arange(0, 360, AZ_STEP)

    # Build windows
    starts = np.arange(0, n_total - win_len, step_len)
    times  = (starts + win_len / 2) / sr   # centre time of each window

    print(f"  {len(starts)} windows × {len(azimuths)} azimuths ...")
    heatmap = np.zeros((len(starts), len(azimuths)))

    for i, start in enumerate(starts):
        win = data[start: start + win_len]
        channels = [win[:, m] for m in range(win.shape[1])]
        n_fft = 1 << (win_len - 1).bit_length()
        for j, az in enumerate(azimuths):
            heatmap[i, j] = gcc_phat_power_at_az(channels, sr, az, n_fft)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(starts)}", flush=True)

    # Normalise each time row to [0,1] so overall loudness doesn't dominate
    row_min = heatmap.min(axis=1, keepdims=True)
    row_max = heatmap.max(axis=1, keepdims=True)
    heatmap_norm = (heatmap - row_min) / (row_max - row_min + 1e-12)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    # Main heatmap
    ax = axes[0]
    im = ax.imshow(
        heatmap_norm.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, 360],
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_ylabel("Azimuth (°)", fontsize=12)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["0° Front", "90° Left", "180° Back", "270° Right", "360°"])
    ax.set_title("Time–Azimuth SRP-PHAT heatmap — mixture.wav\n"
                 "(row-normalised: bright = dominant direction at that moment)",
                 fontsize=13)
    plt.colorbar(im, ax=ax, label="Normalised SRP-PHAT power")

    # Marginal: sum over time → overall SRP-PHAT spectrum
    ax2 = axes[1]
    marginal = heatmap.mean(axis=0)
    ax2.plot(azimuths, marginal, color="steelblue", linewidth=1.2)
    ax2.fill_between(azimuths, marginal.min(), marginal, alpha=0.25, color="steelblue")
    ax2.set_xlabel("Azimuth (°)", fontsize=12)
    ax2.set_ylabel("Mean SRP power", fontsize=11)
    ax2.set_title("Time-averaged SRP-PHAT spectrum (peaks = speaker directions)")
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])

    plt.tight_layout()
    out_path = os.path.join(OUT, "doa_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
