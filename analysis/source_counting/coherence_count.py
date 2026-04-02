"""
Coherence-based source counting on example_mixture.wav.

This script uses two related views of the same multichannel scene:

1. Short-block spatial coherence to estimate how many source components are
   dominant at a given moment.
2. Blockwise SRP-PHAT peak accumulation to estimate how many distinct source
   directions persist across the whole recording.

The first quantity is a soft instantaneous count. The second is the scene-level
talker count used as the final estimate.

Outputs saved to analysis/source_counting/results/
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

LF, LR, RF, RR = 0, 1, 2, 3
ALL_PAIRS = [(LF, LR), (LF, RF), (LF, RR), (LR, RF), (LR, RR), (RF, RR)]

NFFT = 1024
HOP = 256
BLOCK_FRAMES = 24
BLOCK_HOP_FRAMES = BLOCK_FRAMES // 2
FREQ_RANGE_HZ = [300.0, 3500.0]
AZIMUTHS_DEG = np.linspace(0, 360, 720, endpoint=False)
BLOCK_PEAKS = 2
BLOCK_PEAK_THRESHOLD_PERCENTILE = 85.0
MIN_PEAK_SEPARATION_DEG = 35.0
FINAL_PEAK_RELATIVE_HEIGHT = 0.15
SMOOTH_KERNEL = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
SMOOTH_KERNEL /= SMOOTH_KERNEL.sum()


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def circular_distance_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def nearest_cardinal(angle_deg):
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda ref: circular_distance_deg(angle_deg, ref))


def select_distinct_peaks(power, azimuths_deg, min_separation_deg, threshold=None, max_peaks=None):
    prev_vals = np.roll(power, 1)
    next_vals = np.roll(power, -1)
    candidate_idx = np.where((power > prev_vals) & (power >= next_vals))[0]

    if threshold is not None:
        candidate_idx = candidate_idx[power[candidate_idx] >= threshold]

    if len(candidate_idx) == 0:
        candidate_idx = np.array([int(np.argmax(power))])

    ordered = candidate_idx[np.argsort(power[candidate_idx])[::-1]]
    chosen = []
    for idx in ordered:
        angle = azimuths_deg[idx]
        if all(circular_distance_deg(angle, azimuths_deg[j]) >= min_separation_deg for j in chosen):
            chosen.append(int(idx))
        if max_peaks is not None and len(chosen) == max_peaks:
            break

    if not chosen:
        chosen = [int(np.argmax(power))]

    return np.array(sorted(chosen, key=lambda idx: azimuths_deg[idx]), dtype=int)


def circular_smooth(values, kernel):
    pad = len(kernel) // 2
    wrapped = np.concatenate([values[-pad:], values, values[:pad]])
    return np.convolve(wrapped, kernel, mode="same")[pad:-pad]


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa_samples(mic_pos, ch_a, ch_b, azimuths_deg, sr, c):
    phi = np.deg2rad(azimuths_deg)
    tau = (
        (mic_pos[ch_a, 0] - mic_pos[ch_b, 0]) * np.sin(phi)
        + (mic_pos[ch_a, 1] - mic_pos[ch_b, 1]) * np.cos(phi)
    ) / c
    return tau * sr


def block_srp_phat_spectrum(block, pair_delay_samples):
    n_samples = block.shape[0]
    n_fft = 1 << (n_samples - 1).bit_length()
    power = np.zeros(len(AZIMUTHS_DEG), dtype=np.float64)

    for pair, delay_samples in pair_delay_samples.items():
        gcc = gcc_phat(block[:, pair[0]], block[:, pair[1]], n_fft)
        lag_idx = delay_samples + n_fft // 2
        lo = np.floor(lag_idx).astype(int)
        hi = lo + 1
        frac = lag_idx - lo
        lo = np.clip(lo, 0, n_fft - 1)
        hi = np.clip(hi, 0, n_fft - 1)
        power += (1.0 - frac) * gcc[lo] + frac * gcc[hi]

    return (power - power.min()) / (power.max() - power.min() + 1e-12)


def coherence_effective_rank(X_block):
    norm = np.sqrt(np.sum(np.abs(X_block) ** 2, axis=0, keepdims=True)) + 1e-12
    Y = X_block / norm
    Rc = np.einsum("mft,nft->mn", Y, np.conj(Y)) / (X_block.shape[1] * X_block.shape[2])
    eigenvalues = np.sort(np.linalg.eigvalsh(Rc).real)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    participation_ratio = (eigenvalues.sum() ** 2) / (np.square(eigenvalues).sum() + 1e-12)
    return participation_ratio, eigenvalues


with open(GEO_PATH) as f:
    geo = json.load(f)

D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
MIC_POS = np.array([
    [D / 2, L / 2],    # LF
    [D / 2, -L / 2],   # LR
    [-D / 2, L / 2],   # RF
    [-D / 2, -L / 2],  # RR
], dtype=np.float64)

pair_delay_samples = {
    pair: predicted_tdoa_samples(MIC_POS, pair[0], pair[1], AZIMUTHS_DEG, 44100, C)
    for pair in ALL_PAIRS
}

print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s"
)

pair_delay_samples = {
    pair: predicted_tdoa_samples(MIC_POS, pair[0], pair[1], AZIMUTHS_DEG, sr, C)
    for pair in ALL_PAIRS
}

print(
    "\nRunning coherence-based source counting "
    f"(nfft={NFFT}, hop={HOP}, block_frames={BLOCK_FRAMES}) ..."
)
X_all = []
for ch in range(data.shape[1]):
    freqs_hz, _, Z = stft(
        data[:, ch],
        fs=sr,
        window="hann_periodic",
        nperseg=NFFT,
        noverlap=NFFT - HOP,
        boundary=None,
        padded=False,
    )
    X_all.append(Z)
X_all = np.stack(X_all, axis=0)
valid_freq = (freqs_hz >= FREQ_RANGE_HZ[0]) & (freqs_hz <= FREQ_RANGE_HZ[1])

effective_ranks = []
dominant_counts = []
block_times_sec = []
direction_hist = np.zeros(len(AZIMUTHS_DEG), dtype=np.float64)
block_samples = (BLOCK_FRAMES - 1) * HOP + NFFT

for start in range(0, X_all.shape[2] - BLOCK_FRAMES + 1, BLOCK_HOP_FRAMES):
    X_block = X_all[:, valid_freq, start : start + BLOCK_FRAMES]
    eff_rank, eigenvalues = coherence_effective_rank(X_block)
    effective_ranks.append(eff_rank)
    dominant_counts.append(int(np.clip(np.rint(eff_rank), 1, data.shape[1])))
    block_times_sec.append((start * HOP + 0.5 * block_samples) / sr)

    s0 = start * HOP
    s1 = min(data.shape[0], s0 + block_samples)
    block = data[s0:s1, :]
    block_power = block_srp_phat_spectrum(block, pair_delay_samples)
    peak_threshold = np.percentile(block_power, BLOCK_PEAK_THRESHOLD_PERCENTILE)
    block_peak_idx = select_distinct_peaks(
        block_power,
        AZIMUTHS_DEG,
        min_separation_deg=MIN_PEAK_SEPARATION_DEG,
        threshold=peak_threshold,
        max_peaks=BLOCK_PEAKS,
    )
    direction_hist[block_peak_idx] += block_power[block_peak_idx] * eff_rank

effective_ranks = np.asarray(effective_ranks, dtype=np.float64)
dominant_counts = np.asarray(dominant_counts, dtype=int)
block_times_sec = np.asarray(block_times_sec, dtype=np.float64)

direction_hist_smooth = circular_smooth(direction_hist, SMOOTH_KERNEL)
final_threshold = max(
    np.percentile(direction_hist_smooth, 85.0),
    direction_hist_smooth.max() * FINAL_PEAK_RELATIVE_HEIGHT,
)
scene_peak_idx = select_distinct_peaks(
    direction_hist_smooth,
    AZIMUTHS_DEG,
    min_separation_deg=MIN_PEAK_SEPARATION_DEG,
    threshold=final_threshold,
    max_peaks=None,
)
scene_angles_deg = AZIMUTHS_DEG[scene_peak_idx]
scene_cardinals = [nearest_cardinal(angle) for angle in scene_angles_deg]
scene_count = len(scene_angles_deg)

rounded_hist = {
    str(int(k)): int((dominant_counts == k).sum())
    for k in sorted(np.unique(dominant_counts))
}

print(f"  Processed {len(block_times_sec)} overlapping blocks")
print(
    f"  Instantaneous effective rank: median={np.median(effective_ranks):.2f}, "
    f"p90={np.percentile(effective_ranks, 90):.2f}, max={effective_ranks.max():.2f}"
)
print(f"  Rounded dominant-count histogram: {rounded_hist}")
print(f"  Scene-level source count: {scene_count}")
print(f"  Persistent directions: {[f'{a:.1f}°' for a in scene_angles_deg]}")
print(f"  Nearest cardinals: {[f'{c}°' for c in scene_cardinals]}")

summary = {
    "estimated_scene_source_count": int(scene_count),
    "estimated_scene_angles_deg": [float(a) for a in scene_angles_deg],
    "estimated_scene_cardinals_deg": [int(c) for c in scene_cardinals],
    "median_effective_rank": float(np.median(effective_ranks)),
    "p90_effective_rank": float(np.percentile(effective_ranks, 90)),
    "max_effective_rank": float(effective_ranks.max()),
    "rounded_dominant_count_histogram": rounded_hist,
    "parameters": {
        "nfft": NFFT,
        "hop": HOP,
        "block_frames": BLOCK_FRAMES,
        "block_hop_frames": BLOCK_HOP_FRAMES,
        "freq_range_hz": FREQ_RANGE_HZ,
        "block_peaks": BLOCK_PEAKS,
        "min_peak_separation_deg": MIN_PEAK_SEPARATION_DEG,
    },
}
summary_path = os.path.join(OUT_DIR, "coherence_count_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summary_path}")


fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(block_times_sec, effective_ranks, color="#0f766e", linewidth=1.0)
ax.axhline(np.median(effective_ranks), color="#b45309", linestyle="--", linewidth=1.2,
           label=f"median = {np.median(effective_ranks):.2f}")
for y in [1, 2, 3, 4]:
    ax.axhline(y, color="#d1d5db", linewidth=0.8, zorder=0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Effective source count")
ax.set_title("Coherence-based instantaneous source count — example_mixture.wav")
ax.set_ylim(0.9, 4.1)
ax.legend()
plt.tight_layout()
timeline_path = os.path.join(OUT_DIR, "coherence_count_timeline.png")
plt.savefig(timeline_path, dpi=150)
plt.close()
print(f"Saved: {timeline_path}")


hist_norm = direction_hist_smooth / (direction_hist_smooth.max() + 1e-12)
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(1)
az_rad = np.deg2rad(AZIMUTHS_DEG)
ax.plot(az_rad, hist_norm, linewidth=1.2, color="#7c3aed")
ax.fill(az_rad, hist_norm, alpha=0.25, color="#7c3aed")
for idx in scene_peak_idx:
    pa = np.deg2rad(AZIMUTHS_DEG[idx])
    ax.plot([pa], [hist_norm[idx]], marker="o", markersize=5, color="#14532d")
    ax.text(
        pa,
        min(1.08, hist_norm[idx] + 0.08),
        f"{AZIMUTHS_DEG[idx]:.1f}°",
        color="#14532d",
        fontsize=9,
        ha="center",
        va="center",
    )
ax.set_thetagrids(
    [0, 90, 180, 270],
    labels=["Front\n0°", "Left\n90°", "Back\n180°", "Right\n270°"],
    fontsize=10,
    color="#c0392b",
)
ax.set_rticks([])
ax.set_title("Persistent-direction count from blockwise SRP-PHAT", pad=24)
plt.tight_layout()
polar_path = os.path.join(OUT_DIR, "coherence_count_polar.png")
plt.savefig(polar_path, dpi=150)
plt.close()
print(f"Saved: {polar_path}")
