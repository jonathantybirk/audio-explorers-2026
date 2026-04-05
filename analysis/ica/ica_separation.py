"""
Frequency-domain ICA source separation on example_mixture.wav.

This script uses AuxIVA (Auxiliary-function Independent Vector Analysis),
which is the practical frequency-domain ICA variant for reverberant,
convolutive mixtures. That fixes the core problem with the earlier
time-domain FastICA attempt: room mixing is not instantaneous, so real-valued
mixing vectors do not preserve the per-microphone phase delays needed for DoA.

Pipeline:
  1. STFT of the 4-channel mixture.
  2. AuxIVA separation in the STFT domain.
  3. Reconstruct each separated source as:
     - a mono listenable render (projection-back to the average mic), and
     - a 4-channel source image using the complex frequency-domain mixing model.
  4. Run SRP-PHAT on each source image to estimate its DoA.
  5. Save sorted WAVs and diagnostic plots.

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
_WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav"),
}
_parser = argparse.ArgumentParser()
_parser.add_argument("--wav", choices=["example", "mixture"], default="example")
_args = _parser.parse_args()
WAV_KEY = _args.wav
WAV_PATH = _WAV_PATHS[WAV_KEY]
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

CARDINAL_KEYS = {
    0: "0deg_front",
    90: "90deg_left",
    180: "180deg_back",
    270: "270deg_right",
}

_pfx = "" if WAV_KEY == "example" else "mixture_"

AUXIVA_VARIANTS = [
    {
        "key": "default",
        "label": "default",
        "title": "AuxIVA default",
        "stft_size": 2048,
        "hop_size": 1024,
        "n_iter": 30,
        "stable_prefix": f"{_pfx}ica",
        "section_note": "default hp",
        "update_rule": "IP1",
        "wiener": False,
    },
    {
        "key": "tuned",
        "label": "tuned (example-optimised)",
        "title": "AuxIVA tuned (example-optimised)",
        "stft_size": 2048,
        "hop_size": 512,
        "n_iter": 90,
        "stable_prefix": f"{_pfx}ica_tuned",
        "section_note": "hp tuned on example_mixture",
        "update_rule": "IP1",
        "wiener": False,
    },
    {
        "key": "mixture_tuned",
        "label": "tuned (mixture-optimised)",
        "title": "AuxIVA tuned (mixture-optimised)",
        "stft_size": 512,
        "hop_size": 128,
        "n_iter": 140,
        "stable_prefix": f"{_pfx}ica_mtuned",
        "section_note": "hp tuned directly on mixture.wav (Optuna 80 trials)",
        "update_rule": "IP1",
        "wiener": False,
    },
    {
        "key": "wiener_tuned",
        "label": "Wiener example-tuned",
        "title": "AuxIVA + Wiener (example-tuned)",
        "stft_size": 2048,
        "hop_size": 512,
        "n_iter": 90,
        "stable_prefix": f"{_pfx}ica_wiener",
        "section_note": "example-tuned hp + Wiener post-filter",
        "update_rule": "IP1",
        "wiener": True,
    },
]

# When running on example, skip default (already done) and only run the tuned variants
if WAV_KEY == "example":
    AUXIVA_VARIANTS = [v for v in AUXIVA_VARIANTS if v["key"] != "default"]


# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)

D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [D / 2, L / 2],   # LF
    [D / 2, -L / 2],  # LR
    [-D / 2, L / 2],  # RF
    [-D / 2, -L / 2], # RR
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


# ── AuxIVA separation ─────────────────────────────────────────────────────────
def auxiva_separate(data, stft_size, hop_size, n_iter, update_rule="IP1"):
    analysis_win = pra.hann(stft_size)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop_size)

    X = pra.transform.stft.analysis(data, stft_size, hop_size, win=analysis_win)
    Y, W = pra.bss.auxiva(
        X,
        n_src=data.shape[1],
        n_iter=n_iter,
        proj_back=False,
        return_filters=True,
    )

    # Back-project to the average microphone for a stable mono render.
    gains = pra.bss.projection_back(Y, X.mean(axis=2))
    Y_mono = Y * gains[None, :, :]
    mono_sources = pra.transform.stft.synthesis(Y_mono, stft_size, hop_size, win=synthesis_win)

    # Frequency-domain mixing matrices for source-image reconstruction.
    A = np.linalg.inv(W)

    return X, Y, A, mono_sources, synthesis_win, gains


def apply_wiener_postfilter(Y, gains, synthesis_win, stft_size, hop_size, n_samples, eps=1e-8):
    """Soft ratio-mask Wiener filter applied after AuxIVA separation.

    For each TF bin, the mask for source k is its share of the total power
    across all sources. This suppresses bins where other sources dominate.
    """
    Y_scaled = Y * gains[None, :, :]                         # (T, F, K) projection-back
    power = np.abs(Y_scaled) ** 2                            # (T, F, K)
    total = power.sum(axis=2, keepdims=True) + eps           # (T, F, 1)
    mask = power / total                                     # (T, F, K) ratio mask
    Y_wiener = Y_scaled * mask                               # (T, F, K)
    mono = pra.transform.stft.synthesis(Y_wiener, stft_size, hop_size, win=synthesis_win)
    return mono[:n_samples].real.astype(np.float64)


def reconstruct_source_image_channels(Y, A, source_idx, stft_size, hop_size, synthesis_win, n_samples):
    # Keep the source-specific spatial image on all four microphones.
    image_stft = Y[:, :, source_idx][:, :, None] * A[:, :, source_idx][None, :, :]
    channels = []
    for mic_idx in range(image_stft.shape[2]):
        sig = pra.transform.stft.synthesis(
            image_stft[:, :, mic_idx],
            stft_size,
            hop_size,
            win=synthesis_win,
        )
        channels.append(sig[:n_samples].real.astype(np.float64))
    return channels


def nearest_cardinal_label(angle_deg):
    cardinals = [0, 90, 180, 270]
    best = min(cardinals, key=lambda ref: abs(((angle_deg - ref + 180) % 360) - 180))
    return best


def cardinal_key(angle_deg):
    return CARDINAL_KEYS[angle_deg]


def variant_exact_prefix(variant):
    return "ica_source" if variant["stable_prefix"] == "ica" else f'{variant["stable_prefix"]}_source'


def remove_stale_variant_outputs(variant):
    stable_prefix = variant["stable_prefix"]
    exact_prefix = variant_exact_prefix(variant)

    patterns = [
        f"{exact_prefix}_*deg.wav",
        f"{stable_prefix}_spectrograms.png",
        f"{stable_prefix}_polar.png",
    ]
    patterns.extend(f"{stable_prefix}_{cardinal_name}.wav" for cardinal_name in CARDINAL_KEYS.values())

    for pattern in patterns:
        for stale_path in glob.glob(os.path.join(OUT_DIR, pattern)):
            os.remove(stale_path)


def run_variant(variant, data, sr, azimuths):
    print(
        "Running "
        f'{variant["title"]} '
        f'(4 sources, STFT={variant["stft_size"]}, '
        f'hop={variant["hop_size"]}, iterations={variant["n_iter"]}, '
        f'update_rule={variant.get("update_rule", "IP1")}) ...'
    )
    X, Y, A, mono_sources, synthesis_win, gains = auxiva_separate(
        data,
        stft_size=variant["stft_size"],
        hop_size=variant["hop_size"],
        n_iter=variant["n_iter"],
        update_rule=variant.get("update_rule", "IP1"),
    )
    if variant.get("wiener", False):
        mono_sources = apply_wiener_postfilter(
            Y, gains, synthesis_win, variant["stft_size"], variant["hop_size"], data.shape[0]
        )
    else:
        mono_sources = mono_sources[: data.shape[0], :].real.astype(np.float64)
    print(f"  STFT shape: {X.shape[0]} frames × {X.shape[1]} bins × {X.shape[2]} channels")

    print("\nSeparation diagnostics:")
    print("  Source RMS amplitudes:")
    for k in range(mono_sources.shape[1]):
        rms = np.sqrt(np.mean(mono_sources[:, k] ** 2))
        print(f"    Component {k}: RMS = {rms:.4f}")

    print("  Pairwise cross-correlation (off-diagonal near 0 is good):")
    corr = np.corrcoef(mono_sources.T)
    for i in range(corr.shape[0]):
        row = "    " + "  ".join(f"{corr[i, j]:+.3f}" for j in range(corr.shape[1]))
        print(row)

    print("\nEstimating DoA from AuxIVA source images via SRP-PHAT ...")
    doas = []
    power_grid = []
    cardinal_labels = []

    for k in range(mono_sources.shape[1]):
        image_channels = reconstruct_source_image_channels(
            Y,
            A,
            source_idx=k,
            stft_size=variant["stft_size"],
            hop_size=variant["hop_size"],
            synthesis_win=synthesis_win,
            n_samples=data.shape[0],
        )
        power = srp_phat_spectrum(image_channels, sr, azimuths)
        best_az = float(azimuths[np.argmax(power)])
        doas.append(best_az)
        power_grid.append(power)
        cardinal = nearest_cardinal_label(best_az)
        cardinal_labels.append(cardinal)
        print(f"  Component {k}: estimated DoA = {best_az:.1f}°  →  nearest cardinal {cardinal}°")

    power_grid = np.stack(power_grid, axis=0)
    order = list(np.argsort(doas))

    print("\nSaving separated audio ...")
    remove_stale_variant_outputs(variant)
    stable_prefix = variant["stable_prefix"]
    exact_prefix = variant_exact_prefix(variant)
    for rank, k in enumerate(order):
        az = doas[k]
        exact_path = os.path.join(OUT_DIR, f"{exact_prefix}_{rank + 1}_{az:.0f}deg.wav")
        stable_path = os.path.join(OUT_DIR, f"{stable_prefix}_{cardinal_key(cardinal_labels[k])}.wav")
        save_wav(exact_path, mono_sources[:, k], sr)
        save_wav(stable_path, mono_sources[:, k], sr)

    print("\nPlotting spectrograms ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for rank, k in enumerate(order):
        ax = axes[rank // 2][rank % 2]
        ax.specgram(mono_sources[:, k], Fs=sr, NFFT=512, noverlap=256, cmap="inferno")
        ax.set_title(
            f"{variant['title']} source {rank + 1}  —  est. DoA {doas[k]:.1f}°  "
            f"(nearest {cardinal_labels[k]}°)"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
    plt.suptitle(f"{variant['title']} sources — example_mixture.wav", fontsize=13)
    plt.tight_layout()
    spec_path = os.path.join(OUT_DIR, f"{stable_prefix}_spectrograms.png")
    plt.savefig(spec_path, dpi=150)
    plt.close()
    print(f"  saved  {os.path.relpath(spec_path)}")

    print("Plotting per-source SRP-PHAT polars ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "polar"})
    for rank, k in enumerate(order):
        ax = axes[rank // 2][rank % 2]
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
    plt.suptitle(f"Per-source SRP-PHAT from {variant['title']} source images", fontsize=12)
    plt.tight_layout()
    polar_path = os.path.join(OUT_DIR, f"{stable_prefix}_polar.png")
    plt.savefig(polar_path, dpi=150)
    plt.close()
    print(f"  saved  {os.path.relpath(polar_path)}")

    print("\n════════════════════════════════════════════════════════════════════════")
    print(f"  ICA source separation — example_mixture.wav — {variant['title']}")
    print(
        f"  Method: AuxIVA ({variant['n_iter']} iterations, "
        f"STFT {variant['stft_size']}, hop {variant['hop_size']})"
    )
    print("  Estimated source directions (sorted):")
    for rank, k in enumerate(order):
        print(
            f"    Source {rank + 1}: {doas[k]:.1f}°  "
            f"(nearest cardinal {cardinal_labels[k]}°)"
        )
    print("  Expected: 0°, 90°, 180°, 270°")
    print("════════════════════════════════════════════════════════════════════════")

    return {
        "variant": variant,
        "doas": doas,
        "order": order,
        "cardinal_labels": cardinal_labels,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s\n"
)

azimuths = np.linspace(0, 360, 720, endpoint=False)
variant_results = []
for variant in AUXIVA_VARIANTS:
    print("\n" + "=" * 72)
    variant_results.append(run_variant(variant, data, sr, azimuths))
