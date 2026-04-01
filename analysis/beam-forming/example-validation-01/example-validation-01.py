"""
example-validation-01  --  Example-Only Beamforming Validation Pack
====================================================================

Discovery-phase validation should happen on example_mixture.wav because that is
the only scene with known source positions (0, 90, 180, 270 degrees).

This script generates three comparison families for each known direction:
  1. Closest-mic baseline
  2. Geometry-based delay-and-sum (same geometry as geom-das-01)
  3. Geometry-based broadband MVDR (same geometry as mvdr-01)

Outputs are saved as WAVs plus a short summary file intended for listening and
method comparison. mixture.wav is intentionally not used anywhere here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "data" / "example_mixture.wav"
OUT_DIR = ROOT / "analysis" / "beam-forming" / "example-validation-01"
AUDIO_DIR = OUT_DIR / "audio"

CH_LF, CH_LR, CH_RF, CH_RR = 0, 1, 2, 3
TAU_LR = 29
TAU_FR = 8

WINDOW = "hann"
NPERSEG = 1024
NOVERLAP = 768
DIAG_LOADING = 1e-2
PATTERN_FREQS_HZ = np.array([500, 1000, 2000, 3500])

directions = {
    "0deg_front": 0,
    "90deg_left": 90,
    "180deg_back": 180,
    "270deg_right": 270,
}

# Pure front/back/side sources are tied between two mics; use a fixed
# quadrant-style tie-break so each reference direction maps to one channel.
closest_mic = {
    "0deg_front": ("LF", CH_LF),
    "90deg_left": ("LR", CH_LR),
    "180deg_back": ("RR", CH_RR),
    "270deg_right": ("RF", CH_RF),
}


def to_float64(x):
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float64) / np.iinfo(x.dtype).max
    return x.astype(np.float64)


def unit_vec(deg):
    """Case Figure 2 azimuth: 0° front, positive angles toward listener-left."""
    rad = np.radians(deg)
    return np.array([-np.sin(rad), np.cos(rad)])


mic_pos = np.array(
    [
        [-TAU_LR / 2, TAU_FR / 2],
        [-TAU_LR / 2, -TAU_FR / 2],
        [TAU_LR / 2, TAU_FR / 2],
        [TAU_LR / 2, -TAU_FR / 2],
    ]
)


def steering_delays(deg):
    return (mic_pos[CH_LF] - mic_pos) @ unit_vec(deg)


def steering_vector(freqs_hz, tdoas, fs):
    return np.exp(1j * 2 * np.pi * freqs_hz[:, None] * tdoas[None, :] / fs)


def geom_das(signals, tdoas):
    n = signals.shape[0]
    spectra = np.fft.rfft(signals, axis=0)
    k = np.arange(spectra.shape[0])
    out_spec = np.zeros(spectra.shape[0], dtype=complex)
    for i, tau in enumerate(tdoas):
        out_spec += spectra[:, i] * np.exp(1j * 2 * np.pi * k * tau / n)
    return np.fft.irfft(out_spec / signals.shape[1], n=n)


def multichannel_stft(signals, fs):
    freqs_hz, _, stft_raw = sig.stft(
        signals.T,
        fs=fs,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        padded=True,
        boundary="zeros",
        axis=-1,
    )
    return freqs_hz, np.transpose(stft_raw, (1, 2, 0))


def estimate_covariance(stft_cube):
    n_frames = stft_cube.shape[1]
    return np.einsum("ftm,ftn->fmn", stft_cube, np.conj(stft_cube)) / n_frames


def mvdr_weights(freqs_hz, covariance, tdoas, fs):
    n_mics = covariance.shape[-1]
    steering = steering_vector(freqs_hz, tdoas, fs)
    trace = np.real(np.trace(covariance, axis1=1, axis2=2)) / n_mics
    loaded = covariance + DIAG_LOADING * trace[:, None, None] * np.eye(n_mics)[None, :, :]

    inv_r_d = np.linalg.solve(loaded, steering[:, :, None]).squeeze(-1)
    denom = np.sum(np.conj(steering) * inv_r_d, axis=1, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
    weights = inv_r_d / denom

    distortionless_error = np.mean(np.abs(np.sum(np.conj(weights) * steering, axis=1) - 1.0))
    cond = np.linalg.cond(loaded)
    return weights, distortionless_error, cond


def apply_mvdr(stft_cube, weights, fs):
    output_spec = np.sum(np.conj(weights[:, None, :]) * stft_cube, axis=2)
    _, signal = sig.istft(
        output_spec,
        fs=fs,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        input_onesided=True,
        boundary=True,
    )
    return signal


def save_audio_set(prefix, signals_by_label, fs):
    for label, signal in signals_by_label.items():
        peak = np.max(np.abs(signal)) + 1e-9
        out = (signal / peak * np.iinfo(np.int16).max).astype(np.int16)
        wav.write(AUDIO_DIR / f"{prefix}_{label}.wav", fs, out)
        print(f"Saved: {prefix}_{label}.wav")


def plot_comparison_spectrograms(signals_by_method, fs):
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle(
        "example_mixture.wav validation: closest mic vs geom-DAS vs MVDR",
        fontsize=12,
    )

    method_order = ["closest", "geom_das", "mvdr"]
    for row, method in enumerate(method_order):
        for col, label in enumerate(directions.keys()):
            ax = axes[row, col]
            ax.specgram(
                signals_by_method[method][label],
                Fs=fs,
                NFFT=1024,
                noverlap=512,
                cmap="inferno",
                scale="dB",
            )
            ax.set_ylim(0, 8000)
            if row == 0:
                ax.set_title(label.replace("_", "\n"), fontsize=9)
            if col == 0:
                ax.set_ylabel(method.replace("_", " "), fontsize=9)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    axes[-1, 3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_spectrograms.png", dpi=150)
    print("Saved: comparison_spectrograms.png")


def logmag_pair_corr(signals_by_label, fs):
    feats = []
    for label in directions:
        _, _, Z = sig.stft(
            signals_by_label[label],
            fs=fs,
            window=WINDOW,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
        )
        band = (np.fft.rfftfreq(NPERSEG, d=1 / fs) >= 200) & (np.fft.rfftfreq(NPERSEG, d=1 / fs) <= 4000)
        feat = np.log10(np.abs(Z[band]) + 1e-8)
        feats.append(feat.reshape(-1))
    corr = np.corrcoef(np.stack(feats))
    vals = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            vals.append(corr[i, j])
    return corr, float(np.mean(vals))


def directional_response_table(weights_by_label, freqs_hz, fs, method_name):
    table = {}
    for label, steer_deg in directions.items():
        if method_name == "geom_das":
            steer = steering_delays(steer_deg)
            row = {}
            for probe_label, probe_deg in directions.items():
                delta = steering_delays(probe_deg) - steer
                gain = 0.0
                for hz in PATTERN_FREQS_HZ:
                    phases = np.exp(1j * 2 * np.pi * hz * delta / fs)
                    gain += np.abs(np.mean(phases))
                row[probe_label] = 20 * np.log10(gain / len(PATTERN_FREQS_HZ) + 1e-12)
            table[label] = row
        else:
            freq_idx = np.array([np.argmin(np.abs(freqs_hz - hz)) for hz in PATTERN_FREQS_HZ])
            weights = weights_by_label[label][freq_idx]
            row = {}
            for probe_label, probe_deg in directions.items():
                probe = steering_vector(freqs_hz[freq_idx], steering_delays(probe_deg), fs)
                response = np.abs(np.sum(np.conj(weights) * probe, axis=1))
                row[probe_label] = 20 * np.log10(np.mean(response) + 1e-12)
            table[label] = row
    return table


def write_summary(path, signals_by_method, fs, mvdr_weights_by_label, mvdr_freqs_hz, mvdr_cond, mvdr_err):
    lines = [
        "example-validation-01 summary",
        "=============================",
        "",
        "Intent:",
        "  Validate beamforming methods on example_mixture.wav only.",
        "  This avoids assuming anything about the speaker layout of mixture.wav.",
        "",
        "Known source directions in example_mixture.wav:",
        "  0deg_front, 90deg_left, 180deg_back, 270deg_right",
        "",
        "Closest-mic baselines used for listening:",
    ]
    for label, (mic_label, _) in closest_mic.items():
        lines.append(f"  {label:<14} {mic_label}")

    lines.extend(
        [
            "",
            f"MVDR STFT: window={WINDOW}, nperseg={NPERSEG}, noverlap={NOVERLAP}",
            f"MVDR diagonal loading: {DIAG_LOADING:.3g} * trace(R) / M",
            f"MVDR distortionless mean |w^H d - 1|: {mvdr_err:.3e}",
            f"MVDR loaded covariance median cond: {np.median(mvdr_cond):.2f}",
            "",
        ]
    )

    for method_name in ["closest", "geom_das", "mvdr"]:
        corr, avg_corr = logmag_pair_corr(signals_by_method[method_name], fs)
        lines.append(f"{method_name} average pairwise log-magnitude correlation: {avg_corr:.4f}")
        lines.append(f"{method_name} pairwise log-magnitude correlation matrix:")
        header = f"{'beam':<14} {'0deg_front':>12} {'90deg_left':>12} {'180deg_back':>12} {'270deg_right':>12}"
        lines.append(header)
        for idx, label in enumerate(directions):
            lines.append(
                f"{label:<14} "
                f"{corr[idx, 0]:12.4f} {corr[idx, 1]:12.4f} {corr[idx, 2]:12.4f} {corr[idx, 3]:12.4f}"
            )
        lines.append("")

    geom_table = directional_response_table({}, np.array([]), fs, "geom_das")
    mvdr_table = directional_response_table(mvdr_weights_by_label, mvdr_freqs_hz, fs, "mvdr")

    for title, table in [("Geometry-DAS directional response table (dB)", geom_table), ("MVDR directional response table (dB)", mvdr_table)]:
        lines.append(title)
        lines.append(f"{'beam':<14} {'0deg_front':>12} {'90deg_left':>12} {'180deg_back':>12} {'270deg_right':>12}")
        for label in directions:
            row = table[label]
            lines.append(
                f"{label:<14} "
                f"{row['0deg_front']:12.2f} {row['90deg_left']:12.2f} {row['180deg_back']:12.2f} {row['270deg_right']:12.2f}"
            )
        lines.append("")

    lines.extend(
        [
            "Suggested listening order per direction:",
            "  1. audio/closest_<direction>.wav",
            "  2. audio/geom_das_<direction>.wav",
            "  3. audio/mvdr_<direction>.wav",
            "",
            "Listen for:",
            "  - whether the target voice becomes more dominant",
            "  - whether interferers are actually reduced, not just phase-smeared",
            "  - whether MVDR clearly beats both the closest mic and geom-DAS",
        ]
    )

    path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {path.name}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    fs, example = wav.read(EXAMPLE)
    example = to_float64(example)
    print(f"Sample rate : {fs} Hz")
    print(f"Example     : {example.shape[0] / fs:.2f}s   shape={example.shape}")

    closest = {label: example[:, ch].copy() for label, (_, ch) in closest_mic.items()}
    save_audio_set("closest", closest, fs)

    geom = {label: geom_das(example, steering_delays(deg)) for label, deg in directions.items()}
    save_audio_set("geom_das", geom, fs)

    freqs_hz, stft_cube = multichannel_stft(example, fs)
    covariance = estimate_covariance(stft_cube)
    mvdr = {}
    weights_by_label = {}
    cond = None
    err_values = []
    for label, deg in directions.items():
        weights, distortionless_error, cond = mvdr_weights(freqs_hz, covariance, steering_delays(deg), fs)
        weights_by_label[label] = weights
        mvdr[label] = apply_mvdr(stft_cube, weights, fs)[: example.shape[0]]
        err_values.append(distortionless_error)
    save_audio_set("mvdr", mvdr, fs)

    plot_comparison_spectrograms(
        {"closest": closest, "geom_das": geom, "mvdr": mvdr},
        fs,
    )
    write_summary(
        OUT_DIR / "summary.txt",
        {"closest": closest, "geom_das": geom, "mvdr": mvdr},
        fs,
        weights_by_label,
        freqs_hz,
        cond,
        float(np.mean(err_values)),
    )

    print("\nDone.")
