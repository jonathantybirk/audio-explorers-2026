"""
Mask-based MVDR beamforming using Spatial EM / GMM masks.

This mirrors analysis/mask_mvdr/mask_mvdr.py, but replaces ILRMA masks with
the soft masks produced by the probabilistic Spatial EM prototype.

Current limitation:
  The learned GMM spatial templates can still collapse front/back. When that
  happens, mask construction falls back to SRP-seeded templates, just like the
  direct Spatial EM separator. The resulting beamformer is still useful to test
  whether the soft masks carry value beyond direct masked reconstruction.

Outputs saved to analysis/spatial_em/beamformed/
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

WAV_KEY = "example"   # "example" or "mixture"
_WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav"),
}
WAV_PATH = _WAV_PATHS[WAV_KEY]
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "beamformed")
os.makedirs(OUT_DIR, exist_ok=True)

LF, LR, RF, RR = 0, 1, 2, 3
CHANNEL_LABELS = ["LF", "LR", "RF", "RR"]
ALL_PAIRS = list(itertools.combinations(range(4), 2))

NUM_SOURCES = 4
STFT_SIZE = 1024
HOP_SIZE = 256
LOW_FREQ_RANGE_HZ = [300.0, 900.0]
FULL_FREQ_RANGE_HZ = [250.0, 5000.0]
MIN_BIN_MAG = 5.0e-4
TOP_WEIGHTED_BINS = 60000
PHASE_TEMPERATURE = 6.0
PRIOR_TEMPERATURE = 1.0
REG_COVAR = 1.0e-4
DIAG_LOAD = 1.0e-4
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


def fit_gmm_from_tdoa_bins(X_all, freqs_hz, pair_to_idx, init_angles_deg, mic_pos, c):
    low_mask = (freqs_hz >= LOW_FREQ_RANGE_HZ[0]) & (freqs_hz <= LOW_FREQ_RANGE_HZ[1])
    low_freqs = freqs_hz[low_mask]
    if len(low_freqs) == 0:
        raise RuntimeError("No low-frequency bins left for TDoA fitting.")

    mean_mag = np.mean(np.abs(X_all[:, low_mask, :]), axis=0)
    n_tdoa = len(ALL_PAIRS)
    n_extra = 3
    feature_store = np.zeros((low_freqs.size, X_all.shape[2], n_tdoa + n_extra), dtype=np.float64)

    for pair in ALL_PAIRS:
        cross = X_all[pair[0], low_mask, :] * np.conj(X_all[pair[1], low_mask, :])
        phase = np.angle(cross)
        tau = -phase / (2.0 * np.pi * np.maximum(low_freqs[:, None], 1.0))
        feature_store[:, :, pair_to_idx[pair]] = tau

    low_mag = np.abs(X_all[:, low_mask, :]) + 1e-12
    feature_store[:, :, n_tdoa + 0] = np.log(low_mag[LF] / low_mag[LR])
    feature_store[:, :, n_tdoa + 1] = np.log(low_mag[RF] / low_mag[RR])
    feature_store[:, :, n_tdoa + 2] = np.log((low_mag[LF] + low_mag[LR]) / (low_mag[RF] + low_mag[RR]))

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

    return mean_tdoa, frame_priors


def best_angle_from_tdoa_grid(mean_tau_s, azimuths_deg, mic_pos, c):
    mean_tau_s = np.asarray(mean_tau_s[: len(ALL_PAIRS)], dtype=np.float64)
    errors = []
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

        tau_pair = mean_tdoa[:, pair_to_idx[pair]]
        pred = np.exp(-1j * 2.0 * np.pi * freqs_hz[:, None] * tau_pair[None, :])
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


def reconstruct_image_sources(X_all, masks, sr, n_samples):
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
                    nperseg=STFT_SIZE,
                    noverlap=STFT_SIZE - HOP_SIZE,
                    input_onesided=True,
                    boundary=False,
                )
            channels.append(sig[:n_samples].astype(np.float64))
        image_sources.append(channels)
        mono_sources.append(np.mean(np.stack(channels), axis=0))
    return mono_sources, image_sources


def steering_vector(freqs_hz, doa_deg, mic_pos, c):
    phi = np.deg2rad(doa_deg)
    tau = -(mic_pos[:, 0] * np.sin(phi) + mic_pos[:, 1] * np.cos(phi)) / c
    return np.exp(1j * 2.0 * np.pi * freqs_hz[:, None] * tau[None, :])


with open(GEO_PATH) as f:
    geo = json.load(f)

D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
MIC_POS = np.array([
    [D / 2, L / 2],
    [D / 2, -L / 2],
    [-D / 2, L / 2],
    [-D / 2, -L / 2],
], dtype=np.float64)
PAIR_TO_IDX = {pair: idx for idx, pair in enumerate(ALL_PAIRS)}

print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s\n"
)

print(
    "Running Spatial-EM mask MVDR "
    f"(sources={NUM_SOURCES}, nfft={STFT_SIZE}, hop={HOP_SIZE}) ..."
)
X_all = []
for ch in range(data.shape[1]):
    freqs_hz, _, Z = stft(
        data[:, ch],
        fs=sr,
        window="hann_periodic",
        nperseg=STFT_SIZE,
        noverlap=STFT_SIZE - HOP_SIZE,
        boundary=None,
        padded=False,
    )
    X_all.append(Z)
X_all = np.stack(X_all, axis=0)  # (M, F, T)

azimuths = np.linspace(0, 360, 720, endpoint=False)
scene_power = srp_phat_spectrum([data[:, ch] for ch in range(data.shape[1])], sr, azimuths, MIC_POS, C)
init_angles = select_distinct_peak_angles(
    scene_power,
    azimuths,
    min_separation_deg=INIT_PEAK_MIN_SEPARATION_DEG,
    n_peaks=NUM_SOURCES,
    percentile=INIT_PEAK_PERCENTILE,
)
print(f"  SRP-initialized angles: {[f'{a:.1f}°' for a in init_angles]}")

mean_tdoa, frame_priors = fit_gmm_from_tdoa_bins(X_all, freqs_hz, PAIR_TO_IDX, init_angles, MIC_POS, C)
gmm_angles = [best_angle_from_tdoa_grid(mean_tdoa[k], azimuths, MIC_POS, C) for k in range(NUM_SOURCES)]
print(f"  GMM template angles: {[f'{a:.1f}°' for a in gmm_angles]}")

template_tdoa = mean_tdoa
template_source = "learned_gmm"
if count_distinct_angles(gmm_angles, TEMPLATE_DISTINCTNESS_DEG) < NUM_SOURCES:
    print("  Learned GMM templates collapsed; using SRP-seeded angle templates for masks.")
    template_tdoa = tdoa_templates_from_angles(init_angles, MIC_POS, C, mean_tdoa.shape[1])
    template_source = "srp_seeded_fallback"

masks = build_soft_masks(X_all, freqs_hz, template_tdoa, frame_priors, PAIR_TO_IDX)
masked_mono, image_sources = reconstruct_image_sources(X_all, masks, sr, data.shape[0])

print("\nEstimating DoA from Spatial-EM source images ...")
source_doas = []
for k in range(NUM_SOURCES):
    power = srp_phat_spectrum(image_sources[k], sr, azimuths, MIC_POS, C)
    doa = float(azimuths[np.argmax(power)])
    source_doas.append(doa)
    print(f"  Source {k}: DoA = {doa:.1f}°  →  nearest cardinal {nearest_cardinal(doa)}°")

print("\nApplying mask-MVDR beamforming ...")
X_fmt = X_all.transpose(1, 0, 2)  # (F, M, T)
beamformed = {}
for k in range(NUM_SOURCES):
    doa = source_doas[k]
    card = nearest_cardinal(doa)
    a = steering_vector(freqs_hz, doa, MIC_POS, C)
    mk = masks[k]
    mi = 1.0 - mk

    Rk = np.einsum("ft,mft,nft->fmn", mk, X_all, np.conj(X_all)) / X_all.shape[2]
    Rv = np.einsum("ft,mft,nft->fmn", mi, X_all, np.conj(X_all)) / X_all.shape[2]
    Rv += DIAG_LOAD * np.eye(data.shape[1])[None, :, :]

    Rinv = np.linalg.inv(Rv)
    Rinv_a = np.einsum("fmn,fn->fm", Rinv, a)
    denom = np.einsum("fm,fm->f", a.conj(), Rinv_a).real + 1e-12
    w = Rinv_a / denom[:, None]
    Y_beam = np.einsum("fm,fmt->ft", w.conj(), X_fmt)
    _, out = istft(Y_beam, fs=sr, nperseg=STFT_SIZE, noverlap=STFT_SIZE - HOP_SIZE)
    N = data.shape[0]
    out = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
    beamformed[(k, card, doa)] = out.real.astype(np.float64)
    print(f"  Source {k} ({doa:.1f}°, nearest {card}°) — beamformed")

print("\nSaving beamformed audio ...")
pfx = WAV_KEY
remove_stale_outputs(f"spatial_em_mmvdr_{pfx}_source_*deg.wav")
remove_stale_outputs(f"spatial_em_mmvdr_{pfx}_*deg*.wav")
cardinal_counts = {}
for rank, (meta, signal) in enumerate(sorted(beamformed.items(), key=lambda item: item[0][2]), start=1):
    k, card, doa = meta
    save_wav(os.path.join(OUT_DIR, f"spatial_em_mmvdr_{pfx}_source_{rank}_{doa:.0f}deg.wav"), signal, sr)
    cardinal_counts[card] = cardinal_counts.get(card, 0) + 1
    suffix = "" if cardinal_counts[card] == 1 else f"_{cardinal_counts[card]}"
    save_wav(os.path.join(OUT_DIR, f"spatial_em_mmvdr_{pfx}_{cardinal_key(card)}{suffix}.wav"), signal, sr)

print("\nPlotting spectrograms ...")
ordered = sorted(beamformed.items(), key=lambda item: item[0][2])
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for rank, ((_, card, doa), sig) in enumerate(ordered):
    ax = axes[rank // 2][rank % 2]
    ax.specgram(sig, Fs=sr, NFFT=512, noverlap=256, cmap="cividis")
    ax.set_title(f"Spatial-EM mask MVDR source {rank + 1}  —  {doa:.1f}°  (nearest {card}°)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
plt.suptitle("Spatial-EM mask MVDR beamformed sources — " + os.path.basename(WAV_PATH), fontsize=13)
plt.tight_layout()
spec_path = os.path.join(OUT_DIR, f"spatial_em_mmvdr_{pfx}_spectrograms.png")
plt.savefig(spec_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(spec_path)}")

summary = {
    "wav": os.path.basename(WAV_PATH),
    "method": "Mask-MVDR using Spatial EM / GMM masks",
    "num_sources": NUM_SOURCES,
    "nfft": STFT_SIZE,
    "hop": HOP_SIZE,
    "srp_init_angles_deg": [float(a) for a in init_angles],
    "gmm_template_angles_deg": [float(a) for a in gmm_angles],
    "mask_template_source": template_source,
    "estimated_source_directions_deg": [float(meta[2]) for meta, _ in ordered],
    "estimated_source_cardinals_deg": [int(meta[1]) for meta, _ in ordered],
}
summary_path = os.path.join(OUT_DIR, f"spatial_em_mmvdr_{pfx}_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  saved  {os.path.relpath(summary_path)}")

print("\n════════════════════════════════════════════════════════════════════════")
print("  Spatial-EM mask MVDR — " + os.path.basename(WAV_PATH))
print(f"  STFT={STFT_SIZE}, hop={HOP_SIZE}")
print(f"  Template source: {template_source}")
print(f"  GMM template angles: {[f'{a:.1f}°' for a in gmm_angles]}")
print(f"  Beamformed DoAs: {[f'{meta[2]:.1f}°' for meta, _ in ordered]}")
print("════════════════════════════════════════════════════════════════════════")
