"""
DUET source separation on example_mixture.wav.

This implementation uses a practical DUET-style pipeline:
  1. Pick a 2-channel microphone pair.
  2. Compute attenuation-delay features in the STFT domain.
  3. Cluster time-frequency bins into 4 sources.
  4. Build binary masks from those clusters.
  5. Apply the masks to all 4 microphone channels to reconstruct
     source images for DoA labeling.
  6. Save mono renders and diagnostic plots.

Notes:
  - Classical DUET is a 2-channel method.
  - We use the LF-RR diagonal pair because it gives the most distinct
    delay/attenuation structure on the example scene.
  - The 4-channel source images are only used after the 2-channel DUET
    masks are estimated.
"""

import glob
import itertools
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft
from sklearn.cluster import KMeans

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

LF, LR, RF, RR = 0, 1, 2, 3
CHANNEL_LABELS = ["LF", "LR", "RF", "RR"]
ALL_PAIRS = list(itertools.combinations(range(4), 2))

PAIR = (LF, RR)
NUM_SOURCES = 4
NFFT = 1024
HOP = 256
FREQ_RANGE_HZ = [300.0, 3500.0]
MAX_DELAY_S = 1.0e-3
MAG_THRESHOLD = 5.0e-4
TOP_WEIGHTED_BINS = 40000
RANDOM_STATE = 0


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


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def circular_distance_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def nearest_cardinal(angle_deg):
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda ref: circular_distance_deg(angle_deg, ref))


def cardinal_key(angle_deg):
    return {
        0: "0deg_front",
        90: "90deg_left",
        180: "180deg_back",
        270: "270deg_right",
    }[angle_deg]


def predicted_tdoa(mic_pos, ch_a, ch_b, phi_rad, c):
    return (
        (mic_pos[ch_a, 0] - mic_pos[ch_b, 0]) * np.sin(phi_rad)
        + (mic_pos[ch_a, 1] - mic_pos[ch_b, 1]) * np.cos(phi_rad)
    ) / c


def srp_phat_spectrum(channels, sr, azimuths, mic_pos, c):
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
            tau = predicted_tdoa(mic_pos, ch_a, ch_b, phi, c)
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


def duet_features(X1, X2, freqs_hz):
    mag1 = np.abs(X1)
    mag2 = np.abs(X2)
    ratio = X2 / (X1 + 1e-12)
    log_att = np.log(np.abs(ratio) + 1e-12)
    delay_s = -np.angle(ratio) / (2 * np.pi * np.maximum(freqs_hz, 1e-9))

    valid = (
        (freqs_hz >= FREQ_RANGE_HZ[0])
        & (freqs_hz <= FREQ_RANGE_HZ[1])
        & (mag1 > MAG_THRESHOLD)
        & (mag2 > MAG_THRESHOLD)
        & (np.abs(delay_s) <= MAX_DELAY_S)
        & np.isfinite(log_att)
        & np.isfinite(delay_s)
    )

    return log_att, delay_s, valid, np.sqrt(mag1 * mag2)


def fit_duet_clusters(log_att, delay_s, valid, weights, num_sources):
    features = np.column_stack([log_att[valid], delay_s[valid] * 1000.0])
    sample_weights = weights[valid]

    if len(features) > TOP_WEIGHTED_BINS:
        sample_idx = np.argsort(sample_weights)[-TOP_WEIGHTED_BINS:]
        fit_features = features[sample_idx]
        fit_weights = sample_weights[sample_idx]
    else:
        fit_features = features
        fit_weights = sample_weights

    kmeans = KMeans(n_clusters=num_sources, random_state=RANDOM_STATE, n_init=20)
    kmeans.fit(fit_features, sample_weight=fit_weights)
    centers = kmeans.cluster_centers_

    flat_features = np.column_stack([log_att.ravel(), delay_s.ravel() * 1000.0])
    dist2 = ((flat_features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assignment = np.argmin(dist2, axis=1).reshape(log_att.shape)

    masks = np.zeros((num_sources,) + log_att.shape, dtype=np.float64)
    for k in range(num_sources):
        masks[k] = valid & (assignment == k)

    return centers, masks


def reconstruct_sources(X_all, masks, sr, n_samples):
    mono_sources = []
    image_sources = []
    for k in range(masks.shape[0]):
        channels = []
        for ch in range(X_all.shape[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, sig = istft(
                    X_all[ch] * masks[k],
                    fs=sr,
                    window="hann_periodic",
                    nperseg=NFFT,
                    noverlap=NFFT - HOP,
                    input_onesided=True,
                    boundary=False,
                )
            channels.append(sig[:n_samples].astype(np.float64))
        image_sources.append(channels)
        mono_sources.append(np.mean(np.stack(channels), axis=0))
    return mono_sources, image_sources


# ── Load geometry ─────────────────────────────────────────────────────────────
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


# ── Load mixture and STFT ─────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s\n"
)

print(
    "Running DUET "
    f"(pair={CHANNEL_LABELS[PAIR[0]]}-{CHANNEL_LABELS[PAIR[1]]}, "
    f"nfft={NFFT}, hop={HOP}, sources={NUM_SOURCES}) ..."
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
X_all = np.stack(X_all, axis=0)  # (M, F, T)
freqs_hz = freqs_hz[:, None]

X1 = X_all[PAIR[0]]
X2 = X_all[PAIR[1]]
log_att, delay_s, valid, weights = duet_features(X1, X2, freqs_hz)
print(f"  Valid DUET bins: {int(valid.sum())}")

centers, masks = fit_duet_clusters(log_att, delay_s, valid, weights, NUM_SOURCES)
mono_sources, image_sources = reconstruct_sources(X_all, masks, sr, data.shape[0])

print("\nEstimating DoA from DUET source images via SRP-PHAT ...")
azimuths = np.linspace(0, 360, 720, endpoint=False)
doas = []
power_grid = []
cardinals = []
source_energy = []
for k in range(NUM_SOURCES):
    power = srp_phat_spectrum(image_sources[k], sr, azimuths, MIC_POS, C)
    best_az = float(azimuths[np.argmax(power)])
    doas.append(best_az)
    power_grid.append(power)
    card = nearest_cardinal(best_az)
    cardinals.append(card)
    energy = float(np.sqrt(np.mean(mono_sources[k] ** 2)))
    source_energy.append(energy)
    print(f"  Source {k}: estimated DoA = {best_az:.1f}°  →  nearest cardinal {card}°")

power_grid = np.stack(power_grid, axis=0)
order = list(np.argsort(doas))

print("\nSaving separated audio ...")
for stale_path in glob.glob(os.path.join(OUT_DIR, "duet_source_*deg.wav")):
    os.remove(stale_path)
for stale_path in glob.glob(os.path.join(OUT_DIR, "duet_*deg*.wav")):
    os.remove(stale_path)

cardinal_counts = {}
for rank, k in enumerate(order):
    az = doas[k]
    exact_path = os.path.join(OUT_DIR, f"duet_source_{rank + 1}_{az:.0f}deg.wav")
    save_wav(exact_path, mono_sources[k], sr)

    cardinal = cardinals[k]
    cardinal_counts[cardinal] = cardinal_counts.get(cardinal, 0) + 1
    suffix = "" if cardinal_counts[cardinal] == 1 else f"_{cardinal_counts[cardinal]}"
    stable_path = os.path.join(OUT_DIR, f"duet_{cardinal_key(cardinal)}{suffix}.wav")
    save_wav(stable_path, mono_sources[k], sr)

print("\nPlotting spectrograms ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for rank, k in enumerate(order):
    ax = axes[rank // 2][rank % 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.specgram(mono_sources[k], Fs=sr, NFFT=512, noverlap=256, cmap="plasma")
    ax.set_title(
        f"DUET source {rank + 1}  —  est. DoA {doas[k]:.1f}°  "
        f"(nearest {cardinals[k]}°)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
plt.suptitle(
    f"DUET separated sources — pair {CHANNEL_LABELS[PAIR[0]]}-{CHANNEL_LABELS[PAIR[1]]}",
    fontsize=13,
)
plt.tight_layout()
spec_path = os.path.join(OUT_DIR, "duet_spectrograms.png")
plt.savefig(spec_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(spec_path)}")

print("Plotting attenuation-delay clusters ...")
fig, ax = plt.subplots(figsize=(8.5, 6))
valid_att = log_att[valid]
valid_delay_ms = delay_s[valid] * 1000.0
valid_weights = weights[valid]

if len(valid_att) > 20000:
    scatter_idx = np.argsort(valid_weights)[-20000:]
else:
    scatter_idx = np.arange(len(valid_att))

scatter = ax.scatter(
    valid_att[scatter_idx],
    valid_delay_ms[scatter_idx],
    c=valid_weights[scatter_idx],
    cmap="viridis",
    s=3,
    alpha=0.25,
    linewidths=0,
)
ax.scatter(
    centers[:, 0],
    centers[:, 1],
    color="crimson",
    marker="x",
    s=120,
    linewidths=2,
)
for idx, center in enumerate(centers):
    ax.text(center[0], center[1], f"  C{idx}", color="crimson", fontsize=10, va="center")
ax.set_xlabel("Log attenuation")
ax.set_ylabel("Relative delay (ms)")
ax.set_title(
    f"DUET attenuation-delay clusters — pair {CHANNEL_LABELS[PAIR[0]]}-{CHANNEL_LABELS[PAIR[1]]}"
)
fig.colorbar(scatter, ax=ax, label="DUET bin weight")
plt.tight_layout()
cluster_path = os.path.join(OUT_DIR, "duet_clusters.png")
plt.savefig(cluster_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(cluster_path)}")

print("Plotting per-source SRP-PHAT polars ...")
fig, axes = plt.subplots(2, 2, figsize=(10, 9), subplot_kw={"projection": "polar"})
for rank, k in enumerate(order):
    ax = axes[rank // 2][rank % 2]
    curve = power_grid[k]
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-12)
    az_rad = np.deg2rad(azimuths)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.plot(az_rad, curve, linewidth=1.2, color="#7c3aed")
    ax.fill(az_rad, curve, alpha=0.22, color="#7c3aed")
    best = np.deg2rad(doas[k])
    ax.axvline(best, color="crimson", linestyle="--", linewidth=1.0)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"])
    ax.set_rticks([])
    ax.set_title(f"Source {rank + 1}\n{doas[k]:.1f}°", va="bottom")
plt.suptitle("DUET per-source DoA via SRP-PHAT", y=1.02, fontsize=13)
plt.tight_layout()
polar_path = os.path.join(OUT_DIR, "duet_polar.png")
plt.savefig(polar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved  {os.path.relpath(polar_path)}")

print("\n════════════════════════════════════════════════════════════════════════")
print("  DUET source separation — example_mixture.wav")
print(f"  Pair: {CHANNEL_LABELS[PAIR[0]]}-{CHANNEL_LABELS[PAIR[1]]}")
print(f"  STFT: nfft={NFFT}, hop={HOP}")
print(f"  Frequency range: {FREQ_RANGE_HZ[0]:.0f}–{FREQ_RANGE_HZ[1]:.0f} Hz")
print(f"  DoAs (sorted): {[f'{doas[k]:.1f}°' for k in order]}")
print(f"  Nearest cardinals: {[f'{cardinals[k]}°' for k in order]}")
print(f"  Source RMS: {[f'{source_energy[k]:.4f}' for k in order]}")
print("════════════════════════════════════════════════════════════════════════")
