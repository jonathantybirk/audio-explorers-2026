"""
mvdr-01  --  Geometry-Based Broadband MVDR Beamforming
======================================================

Natural follow-up to geom-das-01:
  1. Keep the same 4-mic geometry inferred from the trusted GCC delays
       tau_lr = 29 samples
       tau_fr =  8 samples
  2. Estimate one spatial covariance matrix per STFT bin from mixture.wav
  3. Solve loaded MVDR weights
       w(f) = R(f)^-1 d(f) / (d(f)^H R(f)^-1 d(f))
  4. Reconstruct one beam per cardinal direction: 0, 90, 180, 270 degrees

This is still a blind covariance estimate from the full 4-speaker mixture, so it
will not fully isolate speakers. The goal is to improve on plain delay-and-sum
while staying inside the non-deep pipeline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig


ROOT = Path(__file__).resolve().parents[3]
MIXTURE = ROOT / "data" / "mixture.wav"
OUT_DIR = ROOT / "analysis" / "beam-forming" / "mvdr-01"

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
    "90deg_right": 90,
    "180deg_back": 180,
    "270deg_left": 270,
}


def to_float64(x):
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float64) / np.iinfo(x.dtype).max
    return x.astype(np.float64)


def unit_vec(deg):
    rad = np.radians(deg)
    return np.array([np.sin(rad), np.cos(rad)])


mic_pos = np.array(
    [
        [-TAU_LR / 2, TAU_FR / 2],
        [-TAU_LR / 2, -TAU_FR / 2],
        [TAU_LR / 2, TAU_FR / 2],
        [TAU_LR / 2, -TAU_FR / 2],
    ]
)


def steering_delays(deg):
    """
    Far-field TDOA of each mic relative to LF, in samples.
    Positive tau means the mic arrives after LF and should be advanced.
    """
    return (mic_pos[CH_LF] - mic_pos) @ unit_vec(deg)


def steering_vector(freqs_hz, tdoas, fs):
    return np.exp(1j * 2 * np.pi * freqs_hz[:, None] * tdoas[None, :] / fs)


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


def apply_beamformer(stft_cube, weights, fs):
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


def plot_patterns(freqs_hz, weights_by_label, fs):
    sweep = np.linspace(0, 360, 720, endpoint=False)
    freq_idx = np.array([np.argmin(np.abs(freqs_hz - hz)) for hz in PATTERN_FREQS_HZ])

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), subplot_kw={"projection": "polar"})
    fig.suptitle(
        "MVDR beam patterns  (avg over 500 / 1k / 2k / 3.5kHz)",
        fontsize=12,
    )

    cardinal_response_db = {}

    for ax, (label, steer_deg) in zip(axes, directions.items()):
        weights = weights_by_label[label][freq_idx]
        gains = []
        for theta in sweep:
            probe = steering_vector(freqs_hz[freq_idx], steering_delays(theta), fs)
            response = np.abs(np.sum(np.conj(weights) * probe, axis=1))
            gains.append(np.mean(response))

        gains = np.array(gains)
        gains /= gains.max() + 1e-12

        ax.plot(np.radians(sweep), gains, linewidth=1.2)
        ax.set_title(label.replace("_", "\n"), fontsize=9, pad=10)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
        ax.axvline(np.radians(steer_deg), color="red", linewidth=1, alpha=0.6)

        cardinal_response_db[label] = {}
        for probe_label, probe_deg in directions.items():
            probe = steering_vector(freqs_hz[freq_idx], steering_delays(probe_deg), fs)
            response = np.abs(np.sum(np.conj(weights) * probe, axis=1))
            cardinal_response_db[label][probe_label] = 20 * np.log10(np.mean(response) + 1e-12)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "beam_patterns.png", dpi=150)
    print("Saved: beam_patterns.png")

    return cardinal_response_db


def plot_absolute_spectrograms(signals_by_label, fs):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Broadband MVDR on mixture.wav\n"
        "(geometry steering + loaded full-mixture covariance)",
        fontsize=12,
    )

    for ax, (label, signal) in zip(axes, signals_by_label.items()):
        ax.specgram(signal, Fs=fs, NFFT=1024, noverlap=512, cmap="inferno", scale="dB")
        ax.set_ylim(0, 8000)
        ax.set_ylabel(label.replace("_", "\n"), fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "spectrograms_abs.png", dpi=150)
    print("Saved: spectrograms_abs.png")


def plot_differential_spectrograms(signals_by_label, fs):
    mean_sig = np.mean(list(signals_by_label.values()), axis=0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Differential spectrograms  (beam - mean of all beams)\n"
        "Highlights content unique to each MVDR steering direction",
        fontsize=12,
    )

    for ax, (label, signal) in zip(axes, signals_by_label.items()):
        diff = signal - mean_sig
        ax.specgram(diff, Fs=fs, NFFT=1024, noverlap=512, cmap="RdBu_r", scale="dB")
        ax.set_ylim(0, 8000)
        ax.set_ylabel(label.replace("_", "\n"), fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "spectrograms_diff.png", dpi=150)
    print("Saved: spectrograms_diff.png")


def save_wavs(signals_by_label, fs):
    for label, signal in signals_by_label.items():
        peak = np.max(np.abs(signal)) + 1e-9
        out = (signal / peak * np.iinfo(np.int16).max).astype(np.int16)
        wav.write(OUT_DIR / f"beam_{label}.wav", fs, out)
        print(f"Saved: beam_{label}.wav")


def write_summary(path, cond_by_label, error_by_label, rms_by_label, cardinal_response_db):
    labels = list(directions.keys())
    lines = [
        "mvdr-01 summary",
        "===============",
        "",
        f"STFT: window={WINDOW}, nperseg={NPERSEG}, noverlap={NOVERLAP}",
        f"Diagonal loading: {DIAG_LOADING:.3g} * trace(R) / M",
        "Pattern summary frequencies (Hz): " + ", ".join(str(int(hz)) for hz in PATTERN_FREQS_HZ),
        "",
        "Loaded covariance condition number by steering direction:",
    ]

    for label in labels:
        cond = cond_by_label[label]
        lines.append(
            f"  {label:<14} median={np.median(cond):8.2f}  p95={np.percentile(cond, 95):8.2f}  max={np.max(cond):8.2f}"
        )

    lines.extend(
        [
            "",
            "Distortionless constraint check (mean |w^H d - 1|):",
        ]
    )
    for label in labels:
        lines.append(f"  {label:<14} {error_by_label[label]:.3e}")

    lines.extend(
        [
            "",
            "Output RMS by beam:",
        ]
    )
    for label in labels:
        lines.append(f"  {label:<14} {rms_by_label[label]:.6f}")

    lines.extend(
        [
            "",
            "Cardinal response table (dB, averaged over pattern frequencies):",
            "  rows = steered beam, columns = probe direction",
            "",
            f"{'beam':<14} {'0deg_front':>12} {'90deg_right':>12} {'180deg_back':>12} {'270deg_left':>12}",
        ]
    )
    for label in labels:
        row = cardinal_response_db[label]
        lines.append(
            f"{label:<14} "
            f"{row['0deg_front']:12.2f} "
            f"{row['90deg_right']:12.2f} "
            f"{row['180deg_back']:12.2f} "
            f"{row['270deg_left']:12.2f}"
        )

    path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fs, mixture = wav.read(MIXTURE)
    mixture = to_float64(mixture)

    print(f"Sample rate : {fs} Hz")
    print(f"Mixture     : {mixture.shape[0] / fs:.2f}s   shape={mixture.shape}")
    print(f"STFT        : nperseg={NPERSEG}, noverlap={NOVERLAP}, window={WINDOW}")
    print(f"Diag load   : {DIAG_LOADING:.3g} * trace(R) / M")

    print("\n-- Geometry-computed steering delays --------------------------------")
    print(f"{'Direction':<14} {'tau_LF':>10} {'tau_LR':>10} {'tau_RF':>10} {'tau_RR':>10}")
    print("-" * 60)
    steering = {}
    for label, deg in directions.items():
        tdoa = steering_delays(deg)
        steering[label] = tdoa
        print(
            f"{label:<14} "
            f"{tdoa[0]:+10.1f} {tdoa[1]:+10.1f} {tdoa[2]:+10.1f} {tdoa[3]:+10.1f}"
        )

    freqs_hz, stft_cube = multichannel_stft(mixture, fs)
    covariance = estimate_covariance(stft_cube)

    weights_by_label = {}
    signals_by_label = {}
    cond_by_label = {}
    error_by_label = {}
    rms_by_label = {}

    for label, tdoa in steering.items():
        weights, distortionless_error, cond = mvdr_weights(freqs_hz, covariance, tdoa, fs)
        beam = apply_beamformer(stft_cube, weights, fs)[: mixture.shape[0]]

        weights_by_label[label] = weights
        signals_by_label[label] = beam
        cond_by_label[label] = cond
        error_by_label[label] = distortionless_error
        rms_by_label[label] = np.sqrt(np.mean(beam**2))

        print(
            f"{label:<14} rms={rms_by_label[label]:.5f}  "
            f"median-cond={np.median(cond):.2f}  "
            f"distortionless={distortionless_error:.2e}"
        )

    cardinal_response_db = plot_patterns(freqs_hz, weights_by_label, fs)
    plot_absolute_spectrograms(signals_by_label, fs)
    plot_differential_spectrograms(signals_by_label, fs)
    save_wavs(signals_by_label, fs)
    write_summary(
        OUT_DIR / "summary.txt",
        cond_by_label,
        error_by_label,
        rms_by_label,
        cardinal_response_db,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
