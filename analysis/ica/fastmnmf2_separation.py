"""
FastMNMF2 source separation on example_mixture.wav.

FastMNMF2 (Fast Multichannel NMF with Jointly-Diagonalizable Spatial Covariance
Matrices) is the next step beyond ILRMA. The key difference:

  - ILRMA assumes the mixing matrix diagonalises the spatial covariance per
    source, but estimates W by gradient descent on a likelihood that is only
    locally consistent with that assumption.
  - FastMNMF2 directly parameterises the spatial covariance via a shared
    diagonaliser Q (one matrix, all frequencies) plus per-source diagonal
    components. This is a stricter but more principled spatial model, and the
    update equations are closed-form EM steps — no convergence instability.

In practice FastMNMF2 tends to outperform ILRMA when:
  - The scene is reverberant (SCMs don't factorise cleanly).
  - Data is limited (our 21s clip) — fewer spatial degrees of freedom.
  - Sources are close in angle (front/back pair).

The downside: no demixing matrix W is returned — only source images. So DoA
is estimated directly from the returned 4-channel source images via SRP-PHAT.

Outputs saved to analysis/ica/separated/
"""

import argparse
import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_parser = argparse.ArgumentParser()
_parser.add_argument("--wav", choices=["example", "mixture"], default="example")
_args = _parser.parse_args()
WAV_KEY = _args.wav

_WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav"),
}
WAV_PATH = _WAV_PATHS[WAV_KEY]
_pfx = "" if WAV_KEY == "example" else "mixture_"
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

CARDINAL_KEYS = {
    0: "0deg_front",
    90: "90deg_left",
    180: "180deg_back",
    270: "270deg_right",
}

_n_src_default = 4 if WAV_KEY == "example" else 5

# For mixture we sweep n_src=5..8; for example we only run n_src=4 (known ground truth)
if WAV_KEY == "example":
    _nsrc_sweep = [4]
else:
    _nsrc_sweep = [5, 6, 7, 8]

VARIANTS = []

# ── Default hyperparams (one variant per n_src) ──────────────────────────────
for _n in _nsrc_sweep:
    _label = f"n_src={_n}" if WAV_KEY == "mixture" else ""
    VARIANTS.append({
        "key":        f"fmnmf2{'_n' + str(_n) if WAV_KEY == 'mixture' else ''}",
        "title":      f"FastMNMF2 default{(' ' + _label) if _label else ''}",
        "stft_size":  2048,
        "hop_size":   1024,
        "n_iter":     50,
        "n_components": 8,
        "n_src":      _n,
        "prefix":     f"{_pfx}fmnmf2{'_n' + str(_n) if WAV_KEY == 'mixture' else ''}",
    })

# ── Tuned hyperparams (one variant per n_src) ────────────────────────────────
for _n in _nsrc_sweep:
    _label = f"n_src={_n}" if WAV_KEY == "mixture" else ""
    VARIANTS.append({
        "key":        f"fmnmf2_tuned{'_n' + str(_n) if WAV_KEY == 'mixture' else ''}",
        "title":      f"FastMNMF2 tuned{(' ' + _label) if _label else ''}",
        "stft_size":  2048,
        "hop_size":   512,
        "n_iter":     100,
        "n_components": 6,
        "n_src":      _n,
        "prefix":     f"{_pfx}fmnmf2_tuned{'_n' + str(_n) if WAV_KEY == 'mixture' else ''}",
    })


# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)

D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [D / 2,  L / 2],   # LF
    [D / 2, -L / 2],   # LR
    [-D / 2,  L / 2],  # RF
    [-D / 2, -L / 2],  # RR
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def save_wav(path, signal, sr):
    peak = np.max(np.abs(signal)) + 1e-12
    out = np.clip(signal / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")


# ── Spatial helpers ───────────────────────────────────────────────────────────
def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa(ch_a, ch_b, phi_rad):
    return (
        (MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]) * np.sin(phi_rad)
        + (MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]) * np.cos(phi_rad)
    ) / C


def srp_phat_spectrum(channels, sr, azimuths):
    n = min(len(ch) for ch in channels)
    channels = [ch[:n] for ch in channels]
    n_fft = 1 << (n - 1).bit_length()

    gcc_store = {
        (a, b): gcc_phat(channels[a], channels[b], n_fft)
        for a, b in ALL_PAIRS
    }

    power = np.zeros(len(azimuths), dtype=np.float64)
    for idx, az in enumerate(azimuths):
        phi = np.deg2rad(az)
        score = 0.0
        for ch_a, ch_b in ALL_PAIRS:
            tau = predicted_tdoa(ch_a, ch_b, phi)
            lag_idx = tau * sr + n_fft // 2
            lo = int(np.floor(lag_idx))
            hi = lo + 1
            frac = lag_idx - lo
            lo = np.clip(lo, 0, n_fft - 1)
            hi = np.clip(hi, 0, n_fft - 1)
            gcc = gcc_store[(ch_a, ch_b)]
            score += (1.0 - frac) * gcc[lo] + frac * gcc[hi]
        power[idx] = score

    return power


def nearest_cardinal_label(angle_deg):
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda ref: abs(((angle_deg - ref + 180) % 360) - 180))


# ── FastMNMF2 separation ──────────────────────────────────────────────────────
def fastmnmf2_separate(data, stft_size, hop_size, n_iter, n_components, n_src):
    analysis_win = pra.hann(stft_size)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop_size)

    # (nframes, nfrequencies, nchannels)
    X = pra.transform.stft.analysis(data, stft_size, hop_size, win=analysis_win)

    # Returns (nchannels, nframes, nfrequencies, nsources) when mic_index='all'
    Y_all = pra.bss.fastmnmf2(
        X,
        n_src=n_src,
        n_iter=n_iter,
        n_components=n_components,
        mic_index="all",
    )
    # Y_all: (nchannels, nframes, nfrequencies, nsources)

    n_sources = Y_all.shape[3]
    n_samples = data.shape[0]

    # Synthesise per-source signals
    mono_sources = []
    image_channels_per_source = []

    for k in range(n_sources):
        # Average across mics for a stable mono render
        Y_k_avg = Y_all[:, :, :, k].mean(axis=0)  # (nframes, nfrequencies)
        mono = pra.transform.stft.synthesis(Y_k_avg, stft_size, hop_size, win=synthesis_win)
        mono_sources.append(mono[:n_samples].real.astype(np.float64))

        # Per-mic time-domain signals for DoA via SRP-PHAT
        chans = []
        for mic in range(Y_all.shape[0]):
            sig = pra.transform.stft.synthesis(
                Y_all[mic, :, :, k], stft_size, hop_size, win=synthesis_win
            )
            chans.append(sig[:n_samples].real.astype(np.float64))
        image_channels_per_source.append(chans)

    mono_sources = np.stack(mono_sources, axis=1)  # (n_samples, n_sources)
    return mono_sources, image_channels_per_source


# ── Run variant ───────────────────────────────────────────────────────────────
def run_variant(variant, data, sr, azimuths):
    print(
        f"\nRunning {variant['title']} "
        f"(STFT={variant['stft_size']}, hop={variant['hop_size']}, "
        f"iter={variant['n_iter']}, n_components={variant['n_components']}) ..."
    )

    mono_sources, image_channels_per_source = fastmnmf2_separate(
        data,
        stft_size=variant["stft_size"],
        hop_size=variant["hop_size"],
        n_iter=variant["n_iter"],
        n_components=variant["n_components"],
        n_src=variant["n_src"],
    )
    n_sources = mono_sources.shape[1]

    print("\nSeparation diagnostics:")
    print("  Source RMS amplitudes:")
    for k in range(n_sources):
        rms = np.sqrt(np.mean(mono_sources[:, k] ** 2))
        print(f"    Component {k}: RMS = {rms:.4f}")

    print("  Pairwise cross-correlation (off-diagonal near 0 is good):")
    corr = np.corrcoef(mono_sources.T)
    corr_sum = float(np.sum(np.abs(corr[~np.eye(n_sources, dtype=bool)])))
    for i in range(corr.shape[0]):
        row = "    " + "  ".join(f"{corr[i, j]:+.3f}" for j in range(corr.shape[1]))
        print(row)
    print(f"  Cross-corr sum (lower is better): {corr_sum:.4f}")

    print("\nEstimating DoA via SRP-PHAT on FastMNMF2 source images ...")
    doas = []
    power_grid = []
    cardinal_labels = []

    for k in range(n_sources):
        power = srp_phat_spectrum(image_channels_per_source[k], sr, azimuths)
        best_az = float(azimuths[np.argmax(power)])
        doas.append(best_az)
        power_grid.append(power)
        cardinal = nearest_cardinal_label(best_az)
        cardinal_labels.append(cardinal)
        print(f"  Component {k}: estimated DoA = {best_az:.1f}°  →  nearest cardinal {cardinal}°")

    power_grid = np.stack(power_grid, axis=0)
    order = list(np.argsort(doas))

    # ── Save audio ────────────────────────────────────────────────────────────
    prefix = variant["prefix"]
    for stale in glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.wav")):
        os.remove(stale)
    for stale in glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.png")):
        os.remove(stale)

    print("\nSaving separated audio ...")
    for rank, k in enumerate(order):
        az = doas[k]
        cardinal = CARDINAL_KEYS.get(cardinal_labels[k], f"{cardinal_labels[k]}deg")
        exact_path = os.path.join(OUT_DIR, f"{prefix}_source_{rank + 1}_{az:.0f}deg.wav")
        stable_path = os.path.join(OUT_DIR, f"{prefix}_{cardinal}.wav")
        save_wav(exact_path, mono_sources[:, k], sr)
        save_wav(stable_path, mono_sources[:, k], sr)

    # ── Spectrogram plot ─────────────────────────────────────────────────────
    ncols = min(3, n_sources)
    nrows = (n_sources + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()
    for rank, k in enumerate(order):
        ax = axes_flat[rank]
        ax.specgram(mono_sources[:, k], Fs=sr, NFFT=512, noverlap=256, cmap="inferno")
        ax.set_title(
            f"{variant['title']} source {rank + 1}  —  est. DoA {doas[k]:.1f}°  "
            f"(nearest {cardinal_labels[k]}°)"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
    for ax in axes_flat[n_sources:]:
        ax.set_visible(False)
    plt.suptitle(f"{variant['title']} sources — {WAV_KEY}_mixture.wav", fontsize=13)
    plt.tight_layout()
    spec_path = os.path.join(OUT_DIR, f"{prefix}_spectrograms.png")
    plt.savefig(spec_path, dpi=150)
    plt.close()
    print(f"  saved  {os.path.relpath(spec_path)}")

    # ── Polar plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), subplot_kw={"projection": "polar"})
    axes_flat = np.array(axes).flatten()
    for rank, k in enumerate(order):
        ax = axes_flat[rank]
        pw = power_grid[k]
        pw_norm = (pw - pw.min()) / (pw.max() - pw.min() + 1e-12)
        az_rad = np.deg2rad(azimuths)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.plot(az_rad, pw_norm, linewidth=1.2, color="steelblue")
        ax.fill(az_rad, pw_norm, alpha=0.25, color="steelblue")
        ax.axvline(np.deg2rad(doas[k]), color="crimson", linewidth=1.5, linestyle="--")
        ax.set_thetagrids(
            [0, 90, 180, 270],
            labels=["0°\nFront", "90°\nLeft", "180°\nBack", "270°\nRight"],
            fontsize=8,
        )
        ax.set_rticks([])
        ax.set_title(
            f"{variant['title']} source {rank + 1}  —  {doas[k]:.1f}°  "
            f"(nearest {cardinal_labels[k]}°)",
            pad=14,
        )
    for ax in axes_flat[n_sources:]:
        ax.set_visible(False)
    plt.suptitle(f"Per-source SRP-PHAT — {variant['title']}", fontsize=12)
    plt.tight_layout()
    polar_path = os.path.join(OUT_DIR, f"{prefix}_polar.png")
    plt.savefig(polar_path, dpi=150)
    plt.close()
    print(f"  saved  {os.path.relpath(polar_path)}")

    print("\n════════════════════════════════════════════════════════════════════════")
    print(f"  {variant['title']} — {WAV_KEY}_mixture.wav")
    print("  Estimated source directions (sorted):")
    for rank, k in enumerate(order):
        print(f"    Source {rank + 1}: {doas[k]:.1f}°  (nearest {cardinal_labels[k]}°)")
    if WAV_KEY == "example":
        print("  Expected: 0°, 90°, 180°, 270°")
    print(f"  Cross-correlation sum: {corr_sum:.4f}")
    print("════════════════════════════════════════════════════════════════════════")


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples  |  {data.shape[1]} channels  |  {sr} Hz  |  {data.shape[0] / sr:.1f} s\n")

azimuths = np.linspace(0, 360, 720, endpoint=False)
for variant in VARIANTS:
    print("\n" + "=" * 72)
    run_variant(variant, data, sr, azimuths)
