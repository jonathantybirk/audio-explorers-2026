"""
ILRMA source separation on a 4-channel mixture.

Independent Low-Rank Matrix Analysis (ILRMA) extends AuxIVA by modelling each
source's spectrogram as a low-rank NMF decomposition.  This gives ILRMA a much
better statistical model of speech — harmonic structure, temporal patterns —
and in practice gives cleaner separation than AuxIVA, especially for the
front/back pair where spatial information (TDoA) is nearly zero.

Reference:
  D. Kitamura et al., "Determined blind source separation unifying independent
  vector analysis and nonnegative matrix factorization," IEEE/ACM TASLP, 2016.

Pipeline:
  1. STFT of the 4-channel mixture.
  2. ILRMA separation in the STFT domain (NMF components per source).
  3. Back-project to reference microphone for a listenable mono render.
  4. Reconstruct 4-channel source images from the frequency-domain mixing model.
  5. Run SRP-PHAT on each source image to estimate its DoA.
  6. Save sorted WAVs and diagnostic plots.

Outputs saved to analysis/ilrma/separated/

For example_mixture.wav this script emits two ILRMA variants:
  - default: original demo hyperparameters
  - tuned: hyperparameters selected by the finished Optuna study

── Hyperparameters ──────────────────────────────────────────────────────────
All hyperparameters are at the top of this file.  Change and rerun to tune.
"""

import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

# ── Recording to process ──────────────────────────────────────────────────────
# "example"  → example_mixture.wav  (known speaker positions, use for calibration)
# "mixture"  → mixture.wav          (unknown positions, re-estimated from signal)
WAV_KEY = "example"

# ── ILRMA hyperparameters ─────────────────────────────────────────────────────
ILRMA_VARIANTS = [
    {
        "key": "default",
        "title": "ILRMA default",
        "prefix": "ilrma",
        "stft": 2048,
        "hop": 1024,
        "iters": 100,
        "n_components": 4,
        "note": "default demo hp",
    },
    {
        "key": "tuned",
        "title": "ILRMA tuned",
        "prefix": "ilrma_tuned",
        "stft": 2048,
        "hop": 512,
        "iters": 110,
        "n_components": 2,
        "note": "tuned on example_mixture.wav",
    },
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav"),
}
WAV_PATH = _WAV_PATHS[WAV_KEY]
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)


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


def remove_stale_outputs(pattern):
    for path in glob.glob(os.path.join(OUT_DIR, pattern)):
        os.remove(path)
        print(f"  removed stale  {os.path.relpath(path)}")


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
    n     = min(len(ch) for ch in channels)
    n_fft = 1 << (n - 1).bit_length()
    channels = [ch[:n] for ch in channels]

    gcc_store = {(a, b): gcc_phat(channels[a], channels[b], n_fft)
                 for a, b in ALL_PAIRS}

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


def nearest_cardinal(angle_deg):
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda ref: abs(((angle_deg - ref + 180) % 360) - 180))


def cardinal_key(angle_deg):
    return {0: "0deg_front", 90: "90deg_left",
            180: "180deg_back", 270: "270deg_right"}[angle_deg]


# ── ILRMA separation ──────────────────────────────────────────────────────────
def ilrma_separate(data, stft_size, hop_size, n_iter, n_components):
    analysis_win  = pra.hann(stft_size)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop_size)

    # (nframes, nfreqs, nchannels)
    X = pra.transform.stft.analysis(data, stft_size, hop_size, win=analysis_win)

    # ILRMA returns (nframes, nfreqs, nsrc) and W (nfreqs, nchannels, nsrc)
    # when return_filters=True and proj_back=False
    Y, W = pra.bss.ilrma(
        X,
        n_src=data.shape[1],
        n_iter=n_iter,
        proj_back=False,
        n_components=n_components,
        return_filters=True,
    )

    # Back-project to average mic for a stable, well-scaled mono render
    gains = pra.bss.projection_back(Y, X.mean(axis=2))
    Y_mono = Y * gains[None, :, :]
    mono_sources = pra.transform.stft.synthesis(Y_mono, stft_size, hop_size, win=synthesis_win)

    # A = mixing matrix, shape (nfreqs, nchannels, nsrc)
    # W from ILRMA is (nfreqs, nchannels, nsrc) — invert along the square axes
    A = np.linalg.inv(W.transpose(0, 2, 1)).transpose(0, 2, 1)

    return X, Y, A, mono_sources, synthesis_win


def reconstruct_source_image_channels(Y, A, source_idx, stft_size, hop_size, synthesis_win, n_samples):
    # Y[:, :, k] * A[:, :, k] broadcasts to (nframes, nfreqs, nchannels)
    image_stft = Y[:, :, source_idx][:, :, None] * A[:, :, source_idx][None, :, :]
    channels = []
    for mic_idx in range(image_stft.shape[2]):
        sig = pra.transform.stft.synthesis(
            image_stft[:, :, mic_idx], stft_size, hop_size, win=synthesis_win
        )
        channels.append(sig[:n_samples].real.astype(np.float64))
    return channels


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
      f"|  {sr} Hz  |  {data.shape[0]/sr:.1f} s\n")

azimuths = np.linspace(0, 360, 720, endpoint=False)

for variant in ILRMA_VARIANTS:
    print(
        f"Running {variant['title']} "
        f"(4 sources, STFT={variant['stft']}, hop={variant['hop']}, "
        f"iter={variant['iters']}, NMF_k={variant['n_components']}; "
        f"{variant['note']}) ..."
    )
    X, Y, A, mono_sources, synthesis_win = ilrma_separate(
        data,
        variant["stft"],
        variant["hop"],
        variant["iters"],
        variant["n_components"],
    )
    mono_sources = mono_sources[: data.shape[0], :].real.astype(np.float64)
    print(f"  STFT shape: {X.shape[0]} frames × {X.shape[1]} bins × {X.shape[2]} channels")

    print("\nSeparation diagnostics:")
    print("  Source RMS amplitudes:")
    for k in range(mono_sources.shape[1]):
        print(f"    Component {k}: RMS = {np.sqrt(np.mean(mono_sources[:, k]**2)):.4f}")

    print("  Pairwise cross-correlation (off-diagonal near 0 is good):")
    corr = np.corrcoef(mono_sources.T)
    for i in range(corr.shape[0]):
        print("    " + "  ".join(f"{corr[i, j]:+.3f}" for j in range(corr.shape[1])))

    print("\nEstimating DoA from ILRMA source images via SRP-PHAT ...")
    doas = []
    power_grid = []
    cardinals = []

    for k in range(mono_sources.shape[1]):
        image_channels = reconstruct_source_image_channels(
            Y,
            A,
            source_idx=k,
            stft_size=variant["stft"],
            hop_size=variant["hop"],
            synthesis_win=synthesis_win,
            n_samples=data.shape[0],
        )
        power = srp_phat_spectrum(image_channels, sr, azimuths)
        best_az = float(azimuths[np.argmax(power)])
        doas.append(best_az)
        power_grid.append(power)
        card = nearest_cardinal(best_az)
        cardinals.append(card)
        print(f"  Component {k}: DoA = {best_az:.1f}°  →  nearest cardinal {card}°")

    power_grid = np.stack(power_grid, axis=0)
    order = list(np.argsort(doas))

    print("\nSaving separated audio ...")
    remove_stale_outputs(f"{variant['prefix']}_source_*deg.wav")
    remove_stale_outputs(f"{variant['prefix']}_*deg_*.wav")

    for rank, k in enumerate(order):
        az = doas[k]
        save_wav(
            os.path.join(OUT_DIR, f"{variant['prefix']}_source_{rank+1}_{az:.0f}deg.wav"),
            mono_sources[:, k],
            sr,
        )
        save_wav(
            os.path.join(OUT_DIR, f"{variant['prefix']}_{cardinal_key(cardinals[k])}.wav"),
            mono_sources[:, k],
            sr,
        )

    print("\nPlotting spectrograms ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for rank, k in enumerate(order):
        ax = axes[rank // 2][rank % 2]
        ax.specgram(mono_sources[:, k], Fs=sr, NFFT=512, noverlap=256, cmap="magma")
        ax.set_title(
            f"{variant['title']} source {rank+1}  —  est. DoA {doas[k]:.1f}°  "
            f"(nearest {cardinals[k]}°)"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
    plt.suptitle(f"{variant['title']} sources — " + os.path.basename(WAV_PATH), fontsize=13)
    plt.tight_layout()
    spec_path = os.path.join(OUT_DIR, f"{variant['prefix']}_spectrograms.png")
    plt.savefig(spec_path, dpi=150)
    plt.close()
    print(f"  saved  {os.path.relpath(spec_path)}")

    print("Plotting per-source SRP-PHAT polars ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "polar"})
    for rank, k in enumerate(order):
        ax = axes[rank // 2][rank % 2]
        pw = power_grid[k]
        pw_n = (pw - pw.min()) / (pw.max() - pw.min() + 1e-12)
        az_rad = np.deg2rad(azimuths)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.plot(az_rad, pw_n, linewidth=1.2, color="#d97706")
        ax.fill(az_rad, pw_n, alpha=0.22, color="#d97706")
        ax.axvline(np.deg2rad(doas[k]), color="crimson", linewidth=1.5, linestyle="--")
        ax.set_thetagrids(
            [0, 90, 180, 270],
            labels=["0°\nFront", "90°\nLeft", "180°\nBack", "270°\nRight"],
            fontsize=8,
        )
        ax.set_rticks([])
        ax.set_title(
            f"{variant['title']} source {rank+1}  —  {doas[k]:.1f}°  "
            f"(nearest {cardinals[k]}°)",
            pad=14,
        )
    plt.suptitle(
        f"Per-source SRP-PHAT from {variant['title']} source images — "
        + os.path.basename(WAV_PATH),
        fontsize=12,
    )
    plt.tight_layout()
    polar_path = os.path.join(OUT_DIR, f"{variant['prefix']}_polar.png")
    plt.savefig(polar_path, dpi=150)
    plt.close()
    print(f"  saved  {os.path.relpath(polar_path)}")

    print("\n════════════════════════════════════════════════════════════════════════")
    print(f"  {variant['title']} — " + os.path.basename(WAV_PATH))
    print(
        f"  STFT={variant['stft']}, hop={variant['hop']}, "
        f"iter={variant['iters']}, NMF_k={variant['n_components']}"
    )
    print("  Estimated source directions (sorted):")
    for rank, k in enumerate(order):
        print(f"    Source {rank+1}: {doas[k]:.1f}°  (nearest {cardinals[k]}°)")
    print("════════════════════════════════════════════════════════════════════════\n")
