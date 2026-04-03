"""
Probabilistic spatial EM/GMM separation on a 4-channel mixture.

This is a lightweight spatial-clustering baseline that sits between:
  - global SRP-PHAT peak picking, and
  - fully blind BSS methods like AuxIVA / ILRMA.

Pipeline
────────
1. Compute a 4-channel STFT of the mixture.
2. For low frequencies where phase wrapping is limited, estimate a 6-D TDoA
   feature vector for each active time-frequency bin (one delay per mic pair).
3. Fit a Gaussian mixture model (EM) over those TDoA observations.
4. Convert each component mean TDoA vector into a fullband spatial template.
5. Score every TF bin against those templates to obtain soft source masks.
6. Apply the masks to all 4 channels, reconstruct source images, label by DoA,
   and save listenable renders plus diagnostics.

The fit uses only example_mixture.wav by default. The output is intended as the
first "probabilistic spatial masks" prototype before trying EM-mask MVDR.
"""

import glob
import itertools
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.signal import istft, stft
from sklearn.mixture import GaussianMixture

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# "example"  → example_mixture.wav  (known positions, use as first prototype)
# "mixture"  → mixture.wav          (possible follow-up once the method is stable)
WAV_KEY = "example"

_WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav"),
}
WAV_PATH = _WAV_PATHS[WAV_KEY]
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

LF, LR, RF, RR = 0, 1, 2, 3
CHANNEL_LABELS = ["LF", "LR", "RF", "RR"]
ALL_PAIRS = list(itertools.combinations(range(4), 2))

NUM_SOURCES = 4
NFFT = 1024
HOP = 256
LOW_FREQ_RANGE_HZ = [300.0, 900.0]
FULL_FREQ_RANGE_HZ = [250.0, 5000.0]
MIN_BIN_MAG = 5.0e-4
TOP_WEIGHTED_BINS = 60000
PHASE_TEMPERATURE = 6.0
PRIOR_TEMPERATURE = 1.0
REG_COVAR = 1.0e-4
RANDOM_STATE = 0
INIT_PEAK_MIN_SEPARATION_DEG = 35.0
INIT_PEAK_PERCENTILE = 75.0
TEMPLATE_DISTINCTNESS_DEG = 20.0


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
    for stale_path in glob.glob(os.path.join(OUT_DIR, pattern)):
        os.remove(stale_path)
        print(f"  removed stale  {os.path.relpath(stale_path)}")


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


def select_distinct_peak_angles(power, azimuths_deg, min_separation_deg, n_peaks, percentile):
    min_dist_bins = max(1, int(min_separation_deg / (360.0 / len(azimuths_deg))))
    peaks, props = find_peaks(
        power,
        distance=min_dist_bins,
        height=np.percentile(power, percentile),
    )
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(power))], dtype=int)
        heights = power[peaks]
    else:
        heights = props["peak_heights"]

    order = np.argsort(heights)[::-1]
    chosen = []
    for idx in order:
        angle = float(azimuths_deg[peaks[idx]])
        if all(circular_distance_deg(angle, other) >= min_separation_deg for other in chosen):
            chosen.append(angle)
        if len(chosen) == n_peaks:
            break

    if len(chosen) < n_peaks:
        ranked = np.argsort(power)[::-1]
        for idx in ranked:
            angle = float(azimuths_deg[idx])
            if all(circular_distance_deg(angle, other) >= min_separation_deg for other in chosen):
                chosen.append(angle)
            if len(chosen) == n_peaks:
                break

    return sorted(chosen)


def count_distinct_angles(angles_deg, min_separation_deg):
    chosen = []
    for angle in sorted(float(a) for a in angles_deg):
        if all(circular_distance_deg(angle, other) >= min_separation_deg for other in chosen):
            chosen.append(angle)
    return len(chosen)


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


def fit_gmm_from_tdoa_bins(X_all, freqs_hz, pair_to_idx, init_angles_deg, mic_pos, c):
    low_mask = (freqs_hz >= LOW_FREQ_RANGE_HZ[0]) & (freqs_hz <= LOW_FREQ_RANGE_HZ[1])
    low_freqs = freqs_hz[low_mask]
    if len(low_freqs) == 0:
        raise RuntimeError("No low-frequency bins left for TDoA fitting.")

    mean_mag = np.mean(np.abs(X_all[:, low_mask, :]), axis=0)  # (F_low, T)
    n_tdoa = len(ALL_PAIRS)
    n_extra = 3
    feature_store = np.zeros((low_freqs.size, X_all.shape[2], n_tdoa + n_extra), dtype=np.float64)

    for pair in ALL_PAIRS:
        cross = X_all[pair[0], low_mask, :] * np.conj(X_all[pair[1], low_mask, :])
        phase = np.angle(cross)
        tau = -phase / (2.0 * np.pi * np.maximum(low_freqs[:, None], 1.0))
        feature_store[:, :, pair_to_idx[pair]] = tau

    low_mag = np.abs(X_all[:, low_mask, :]) + 1e-12
    feature_store[:, :, n_tdoa + 0] = np.log(low_mag[LF] / low_mag[LR])  # left front/back
    feature_store[:, :, n_tdoa + 1] = np.log(low_mag[RF] / low_mag[RR])  # right front/back
    feature_store[:, :, n_tdoa + 2] = np.log((low_mag[LF] + low_mag[LR]) / (low_mag[RF] + low_mag[RR]))  # left/right

    valid = np.isfinite(feature_store).all(axis=2) & (mean_mag > MIN_BIN_MAG)
    valid_idx = np.flatnonzero(valid.ravel())
    if len(valid_idx) < NUM_SOURCES * 100:
        raise RuntimeError(
            f"Not enough valid low-frequency TF bins for EM/GMM fit ({len(valid_idx)} found)."
        )

    flat_features = feature_store.reshape(-1, n_tdoa + n_extra)
    flat_weights = mean_mag.ravel()
    valid_features = flat_features[valid_idx]
    valid_weights = flat_weights[valid_idx]

    if len(valid_idx) > TOP_WEIGHTED_BINS:
        fit_idx = valid_idx[np.argsort(valid_weights)[-TOP_WEIGHTED_BINS:]]
    else:
        fit_idx = valid_idx

    fit_features = flat_features[fit_idx]
    scale_mean = fit_features.mean(axis=0)
    scale_std = fit_features.std(axis=0) + 1e-9
    fit_features_z = (fit_features - scale_mean) / scale_std
    valid_features_z = (valid_features - scale_mean) / scale_std

    init_tdoa = np.array(
        [
            [
                predicted_tdoa(mic_pos, pair[0], pair[1], np.deg2rad(angle), c)
                for pair in ALL_PAIRS
            ]
            for angle in init_angles_deg
        ],
        dtype=np.float64,
    )
    init_features = np.zeros((NUM_SOURCES, n_tdoa + n_extra), dtype=np.float64)
    init_features[:, :n_tdoa] = init_tdoa
    means_init = (init_features - scale_mean[None, :]) / scale_std[None, :]

    gmm = GaussianMixture(
        n_components=NUM_SOURCES,
        covariance_type="full",
        reg_covar=REG_COVAR,
        n_init=6,
        random_state=RANDOM_STATE,
        max_iter=300,
        means_init=means_init,
        weights_init=np.full(NUM_SOURCES, 1.0 / NUM_SOURCES, dtype=np.float64),
    )
    gmm.fit(fit_features_z)
    valid_post = gmm.predict_proba(valid_features_z)
    mean_tdoa = gmm.means_ * scale_std[None, :] + scale_mean[None, :]

    frame_priors = np.full((X_all.shape[2], NUM_SOURCES), 1.0 / NUM_SOURCES, dtype=np.float64)
    low_flat_to_frame = np.tile(np.arange(X_all.shape[2]), low_freqs.size)
    valid_frames = low_flat_to_frame[valid_idx]
    for t in range(X_all.shape[2]):
        mask_t = valid_frames == t
        if np.any(mask_t):
            frame_weights = valid_weights[mask_t]
            post_t = valid_post[mask_t]
            frame_priors[t] = np.average(post_t, axis=0, weights=frame_weights)
    frame_priors /= frame_priors.sum(axis=1, keepdims=True) + 1e-12

    low_assignments = np.full(valid.shape, -1, dtype=int)
    low_assignments.ravel()[valid_idx] = np.argmax(valid_post, axis=1)

    return mean_tdoa, frame_priors, valid, low_assignments, mean_mag, low_mask


def build_soft_masks(X_all, freqs_hz, mean_tdoa, frame_priors, pair_to_idx):
    n_freqs, n_frames = X_all.shape[1], X_all.shape[2]
    masks = np.zeros((NUM_SOURCES, n_freqs, n_frames), dtype=np.float64)
    full_mask = (freqs_hz >= FULL_FREQ_RANGE_HZ[0]) & (freqs_hz <= FULL_FREQ_RANGE_HZ[1])

    logits = np.log(frame_priors[None, :, :] + 1e-8) * PRIOR_TEMPERATURE
    logits = np.repeat(logits, n_freqs, axis=0)

    for pair in ALL_PAIRS:
        Xa = X_all[pair[0]]
        Xb = X_all[pair[1]]
        obs = Xa * np.conj(Xb)
        obs /= np.abs(obs) + 1e-12

        tau_pair = mean_tdoa[:, pair_to_idx[pair]]  # (K,)
        pred = np.exp(-1j * 2.0 * np.pi * freqs_hz[:, None] * tau_pair[None, :])  # (F, K)
        similarity = np.real(obs[:, :, None] * np.conj(pred[:, None, :]))
        logits[full_mask, :, :] += PHASE_TEMPERATURE * similarity[full_mask, :, :]

    logits -= logits.max(axis=2, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True) + 1e-12

    for k in range(NUM_SOURCES):
        masks[k] = probs[:, :, k]

    mean_mag = np.mean(np.abs(X_all), axis=0)
    low_energy = mean_mag < (MIN_BIN_MAG * 0.5)
    if np.any(low_energy):
        for k in range(NUM_SOURCES):
            fallback = np.broadcast_to(frame_priors[:, k], (n_freqs, n_frames))
            masks[k, low_energy] = fallback[low_energy]
        masks /= masks.sum(axis=0, keepdims=True) + 1e-12

    return masks


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


def best_angle_from_tdoa_grid(mean_tau_s, azimuths_deg, mic_pos, c):
    errors = []
    mean_tau_s = np.asarray(mean_tau_s[: len(ALL_PAIRS)], dtype=np.float64)
    for az in azimuths_deg:
        phi = np.deg2rad(az)
        pred = np.array(
            [predicted_tdoa(mic_pos, pair[0], pair[1], phi, c) for pair in ALL_PAIRS],
            dtype=np.float64,
        )
        errors.append(np.mean((mean_tau_s - pred) ** 2))
    return float(azimuths_deg[int(np.argmin(errors))])


def tdoa_templates_from_angles(angles_deg, mic_pos, c, feature_dim):
    templates = np.zeros((len(angles_deg), feature_dim), dtype=np.float64)
    for row, angle in enumerate(angles_deg):
        phi = np.deg2rad(angle)
        templates[row, : len(ALL_PAIRS)] = [
            predicted_tdoa(mic_pos, pair[0], pair[1], phi, c) for pair in ALL_PAIRS
        ]
    return templates


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
PAIR_TO_IDX = {pair: idx for idx, pair in enumerate(ALL_PAIRS)}


print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s\n"
)

print(
    "Running Spatial EM/GMM "
    f"(sources={NUM_SOURCES}, nfft={NFFT}, hop={HOP}, "
    f"low-fit={LOW_FREQ_RANGE_HZ[0]:.0f}-{LOW_FREQ_RANGE_HZ[1]:.0f} Hz) ..."
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

raw_channels = [data[:, ch] for ch in range(data.shape[1])]
azimuths = np.linspace(0, 360, 720, endpoint=False)
scene_power = srp_phat_spectrum(raw_channels, sr, azimuths, MIC_POS, C)
init_angles = select_distinct_peak_angles(
    scene_power,
    azimuths,
    min_separation_deg=INIT_PEAK_MIN_SEPARATION_DEG,
    n_peaks=NUM_SOURCES,
    percentile=INIT_PEAK_PERCENTILE,
)
print(f"  SRP-initialized angles: {[f'{a:.1f}°' for a in init_angles]}")

mean_tdoa, frame_priors, low_valid, low_assignments, low_mean_mag, low_mask = fit_gmm_from_tdoa_bins(
    X_all, freqs_hz, PAIR_TO_IDX, init_angles, MIC_POS, C
)
print(f"  Low-frequency fit bins: {int(low_valid.sum())}")
print(f"  Mean frame priors shape: {frame_priors.shape}")

gmm_angles = [
    best_angle_from_tdoa_grid(mean_tdoa[k], azimuths, MIC_POS, C)
    for k in range(NUM_SOURCES)
]
print("  GMM component angles from mean TDoA vectors:")
for k, ang in enumerate(gmm_angles):
    print(f"    Component {k}: {ang:.1f}°")

template_tdoa = mean_tdoa
template_source = "learned_gmm"
if count_distinct_angles(gmm_angles, TEMPLATE_DISTINCTNESS_DEG) < NUM_SOURCES:
    print(
        "  Learned GMM templates collapsed spatially; "
        "using SRP-seeded angle templates for mask construction."
    )
    template_tdoa = tdoa_templates_from_angles(init_angles, MIC_POS, C, mean_tdoa.shape[1])
    template_source = "srp_seeded_fallback"

masks = build_soft_masks(X_all, freqs_hz, template_tdoa, frame_priors, PAIR_TO_IDX)
mono_sources, image_sources = reconstruct_sources(X_all, masks, sr, data.shape[0])

print("\nEstimating DoA from Spatial EM source images via SRP-PHAT ...")
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
    print(
        f"  Source {k}: estimated DoA = {best_az:.1f}°  →  nearest cardinal {card}°"
    )

power_grid = np.stack(power_grid, axis=0)
corr = np.corrcoef(np.stack(mono_sources, axis=0))
off_diag = ~np.eye(corr.shape[0], dtype=bool)
cross_corr_sum = float(np.sum(np.abs(corr[off_diag])))
order = list(np.argsort(doas))

print("\nSaving separated audio ...")
pfx = WAV_KEY
remove_stale_outputs(f"spatial_em_{pfx}_source_*deg.wav")
remove_stale_outputs(f"spatial_em_{pfx}_*deg*.wav")

cardinal_counts = {}
for rank, k in enumerate(order):
    az = doas[k]
    exact_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_source_{rank + 1}_{az:.0f}deg.wav")
    save_wav(exact_path, mono_sources[k], sr)

    cardinal = cardinals[k]
    cardinal_counts[cardinal] = cardinal_counts.get(cardinal, 0) + 1
    suffix = "" if cardinal_counts[cardinal] == 1 else f"_{cardinal_counts[cardinal]}"
    stable_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_{cardinal_key(cardinal)}{suffix}.wav")
    save_wav(stable_path, mono_sources[k], sr)

print("\nPlotting spectrograms ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for rank, k in enumerate(order):
    ax = axes[rank // 2][rank % 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.specgram(mono_sources[k], Fs=sr, NFFT=512, noverlap=256, cmap="viridis")
    ax.set_title(
        f"Spatial EM source {rank + 1}  —  est. DoA {doas[k]:.1f}°  "
        f"(nearest {cardinals[k]}°)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
plt.suptitle(f"Spatial EM separated sources — {WAV_KEY}", fontsize=13)
plt.tight_layout()
spec_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_spectrograms.png")
plt.savefig(spec_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(spec_path)}")

print("Plotting low-frequency TDoA clusters ...")
low_freqs = freqs_hz[low_mask]
pair_names = [f"{CHANNEL_LABELS[a]}-{CHANNEL_LABELS[b]}" for a, b in ALL_PAIRS]
pair_a_idx = PAIR_TO_IDX[(LF, RF)]
pair_b_idx = PAIR_TO_IDX[(LF, LR)]

cross_lr = X_all[LF, low_mask, :] * np.conj(X_all[RF, low_mask, :])
tau_lr = -np.angle(cross_lr) / (2.0 * np.pi * np.maximum(low_freqs[:, None], 1.0))
cross_fb = X_all[LF, low_mask, :] * np.conj(X_all[LR, low_mask, :])
tau_fb = -np.angle(cross_fb) / (2.0 * np.pi * np.maximum(low_freqs[:, None], 1.0))

valid_tau_lr = tau_lr[low_valid] * 1000.0
valid_tau_fb = tau_fb[low_valid] * 1000.0
valid_weight = low_mean_mag[low_valid]
valid_label = low_assignments[low_valid]

if len(valid_tau_lr) > 30000:
    scatter_idx = np.argsort(valid_weight)[-30000:]
else:
    scatter_idx = np.arange(len(valid_tau_lr))

fig, ax = plt.subplots(figsize=(8.5, 6))
scatter = ax.scatter(
    valid_tau_lr[scatter_idx],
    valid_tau_fb[scatter_idx],
    c=valid_label[scatter_idx],
    cmap="tab10",
    s=3,
    alpha=0.25,
    linewidths=0,
)
ax.scatter(
    mean_tdoa[:, pair_a_idx] * 1000.0,
    mean_tdoa[:, pair_b_idx] * 1000.0,
    color="crimson",
    marker="x",
    s=120,
    linewidths=2,
)
for idx in range(NUM_SOURCES):
    ax.text(
        mean_tdoa[idx, pair_a_idx] * 1000.0,
        mean_tdoa[idx, pair_b_idx] * 1000.0,
        f"  C{idx}",
        color="crimson",
        fontsize=10,
        va="center",
    )
ax.set_xlabel(f"TDoA {pair_names[pair_a_idx]} (ms)")
ax.set_ylabel(f"TDoA {pair_names[pair_b_idx]} (ms)")
ax.set_title(f"Spatial EM low-frequency TDoA clusters — {WAV_KEY}")
plt.tight_layout()
cluster_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_clusters.png")
plt.savefig(cluster_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(cluster_path)}")

print("Plotting frame priors ...")
times_sec = np.arange(frame_priors.shape[0]) * HOP / sr
fig, ax = plt.subplots(figsize=(11, 4))
for k in range(NUM_SOURCES):
    ax.plot(times_sec, frame_priors[:, k], linewidth=1.2, label=f"Comp {k}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Average posterior")
ax.set_ylim(0.0, 1.0)
ax.set_title(f"Spatial EM frame-level component priors — {WAV_KEY}")
ax.legend(ncol=NUM_SOURCES, fontsize=9)
plt.tight_layout()
priors_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_priors.png")
plt.savefig(priors_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(priors_path)}")

print("Plotting per-source SRP-PHAT polars ...")
fig, axes = plt.subplots(2, 2, figsize=(10, 9), subplot_kw={"projection": "polar"})
for rank, k in enumerate(order):
    ax = axes[rank // 2][rank % 2]
    curve = power_grid[k]
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-12)
    az_rad = np.deg2rad(azimuths)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.plot(az_rad, curve, linewidth=1.2, color="#0f766e")
    ax.fill(az_rad, curve, alpha=0.22, color="#0f766e")
    best = np.deg2rad(doas[k])
    ax.axvline(best, color="crimson", linestyle="--", linewidth=1.0)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"])
    ax.set_rticks([])
    ax.set_title(f"Source {rank + 1}\n{doas[k]:.1f}°", va="bottom")
plt.suptitle(f"Spatial EM per-source DoA via SRP-PHAT — {WAV_KEY}", y=1.02, fontsize=13)
plt.tight_layout()
polar_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_polar.png")
plt.savefig(polar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved  {os.path.relpath(polar_path)}")

summary = {
    "wav": os.path.basename(WAV_PATH),
    "method": "Spatial EM / GMM over low-frequency TDoA bins",
    "num_sources": NUM_SOURCES,
    "nfft": NFFT,
    "hop": HOP,
    "fit_low_freq_range_hz": LOW_FREQ_RANGE_HZ,
    "mask_full_freq_range_hz": FULL_FREQ_RANGE_HZ,
    "srp_init_angles_deg": [float(a) for a in init_angles],
    "gmm_component_angles_deg": [float(a) for a in gmm_angles],
    "mask_template_source": template_source,
    "estimated_source_directions_deg": [float(doas[k]) for k in order],
    "estimated_source_cardinals_deg": [int(cardinals[k]) for k in order],
    "cross_corr_sum": cross_corr_sum,
    "source_rms_sorted": [float(source_energy[k]) for k in order],
}
summary_path = os.path.join(OUT_DIR, f"spatial_em_{pfx}_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  saved  {os.path.relpath(summary_path)}")

print("\n════════════════════════════════════════════════════════════════════════")
print(f"  Spatial EM / GMM source separation — {os.path.basename(WAV_PATH)}")
print(f"  STFT: nfft={NFFT}, hop={HOP}")
print(f"  Low-frequency fit range: {LOW_FREQ_RANGE_HZ[0]:.0f}–{LOW_FREQ_RANGE_HZ[1]:.0f} Hz")
print(f"  Fullband mask range: {FULL_FREQ_RANGE_HZ[0]:.0f}–{FULL_FREQ_RANGE_HZ[1]:.0f} Hz")
print(f"  GMM angles: {[f'{a:.1f}°' for a in gmm_angles]}")
print(f"  DoAs (sorted): {[f'{doas[k]:.1f}°' for k in order]}")
print(f"  Nearest cardinals: {[f'{cardinals[k]}°' for k in order]}")
print(f"  Source RMS: {[f'{source_energy[k]:.4f}' for k in order]}")
print(f"  Cross-correlation sum: {cross_corr_sum:.3f}")
print("════════════════════════════════════════════════════════════════════════")
