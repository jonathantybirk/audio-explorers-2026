"""
geom-das-01  —  Geometry-Based Frequency-Domain Delay-and-Sum
==============================================================

Improvements over gcc-das-01:
  1. Mic positions modelled explicitly from the two trusted measurements:
       τ_lr = 29 samples  (inter-ear distance, high confidence from GCC peak)
       τ_fr =  8 samples  (front-rear spacing, moderate confidence)
     Steering delays are computed analytically via the far-field formula
     τ_i = (p_LF - p_i) · u(θ)  instead of reading noisy GCC peaks.

  2. Frequency-domain beamforming with exact fractional-delay phase shifts
     Y[k] = (1/M) Σ_i  X_i[k] · exp(+j·2π·k·τ_i / N)
     instead of integer sample rolls, avoiding high-frequency phase errors.

Mic layout (positions in samples, x=right, y=front):
  LF: (-τ_lr/2,  +τ_fr/2)    LR: (-τ_lr/2,  -τ_fr/2)
  RF: (+τ_lr/2,  +τ_fr/2)    RR: (+τ_lr/2,  -τ_fr/2)

Azimuth convention (clockwise from front):
  0° = front (+y),  90° = right (+x),  180° = back (−y),  270° = left (−x)
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE    = "/Users/jonathantybirk/Desktop/Audio Explorers 2026"
EXAMPLE = os.path.join(BASE, "data/example_mixture.wav")
MIXTURE = os.path.join(BASE, "data/mixture.wav")
OUT_DIR = os.path.join(BASE, "analysis/beam-forming/geom-das-01")

CH_LF, CH_LR, CH_RF, CH_RR = 0, 1, 2, 3

# ── load ───────────────────────────────────────────────────────────────────
fs, ex = wav.read(EXAMPLE)
_,  mx = wav.read(MIXTURE)
ex = ex.astype(np.float64) / np.iinfo(ex.dtype).max
mx = mx.astype(np.float64) / np.iinfo(mx.dtype).max

print(f"Sample rate : {fs} Hz")
print(f"Example     : {ex.shape[0]/fs:.2f}s")
print(f"Mixture     : {mx.shape[0]/fs:.2f}s")

# ── mic geometry (in samples) ──────────────────────────────────────────────
# Measured from GCC-PHAT on example_mixture (see gcc-das-01):
#   τ_lr = 29 samples  — dominant, unambiguous peak, high confidence
#   τ_fr =  8 samples  — small peak near zero-lag, moderate confidence
TAU_LR = 29   # inter-ear half-distance: each ear ±TAU_LR/2 from centre
TAU_FR =  8   # front-rear half-distance: front/rear mic ±TAU_FR/2 from centre

# Positions as (x, y) in samples.  x = right (+), y = front (+)
mic_pos = np.array([
    [-TAU_LR / 2,  TAU_FR / 2],   # LF
    [-TAU_LR / 2, -TAU_FR / 2],   # LR
    [ TAU_LR / 2,  TAU_FR / 2],   # RF
    [ TAU_LR / 2, -TAU_FR / 2],   # RR
])

def unit_vec(deg):
    """Unit vector pointing FROM listener TOWARD source at azimuth deg."""
    rad = np.radians(deg)
    return np.array([np.sin(rad), np.cos(rad)])   # (x=right, y=front)

def steering_delays(deg):
    """
    Far-field TDOA of each mic relative to LF, in samples.
    τ_i = (p_LF − p_i) · u   →   positive = mic i arrives AFTER LF.
    """
    u = unit_vec(deg)
    return (mic_pos[CH_LF] - mic_pos) @ u   # shape (4,)

# ── compute and print steering vectors ────────────────────────────────────
directions = {"0deg_front": 0, "90deg_right": 90,
              "180deg_back": 180, "270deg_left": 270}

print("\n── Geometry-computed steering delays ─────────────────────────────────")
print(f"{'Direction':<14} {'τ_LF':>12} {'τ_LR':>12} {'τ_RF':>12} {'τ_RR':>12}")
print("─" * 60)
all_tdoas = {}
for label, deg in directions.items():
    td = steering_delays(deg)
    all_tdoas[label] = td
    def fmt(v): return f"{v:+.1f}({v/fs*1000:+.2f}ms)"
    print(f"{label:<14}  {fmt(td[0])}  {fmt(td[1])}  {fmt(td[2])}  {fmt(td[3])}")

# ── frequency-domain delay-and-sum ────────────────────────────────────────
def beamform(signals, tdoas):
    """
    signals : (n_samples, n_mics)
    tdoas   : (n_mics,) fractional delays in samples relative to mic 0
              positive = mic arrives late → advance (apply +phase)
    Returns beamformed signal of length n_samples.
    """
    n = signals.shape[0]
    n_mics = signals.shape[1]
    spectra = np.fft.rfft(signals, axis=0)          # (n_rfft, n_mics)
    k = np.arange(spectra.shape[0])                 # frequency indices
    out_spec = np.zeros(spectra.shape[0], dtype=complex)
    for i, tau in enumerate(tdoas):
        phase = np.exp(+1j * 2 * np.pi * k * tau / n)
        out_spec += spectra[:, i] * phase
    return np.fft.irfft(out_spec / n_mics, n=n)

# ── beam patterns (polar, averaged over speech frequencies) ───────────────
sweep = np.linspace(0, 360, 720, endpoint=False)
f_eval_hz = [500, 1000, 2000, 3500]   # representative speech freqs

fig_pat, axes_pat = plt.subplots(
    1, 4, figsize=(16, 5),
    subplot_kw={"projection": "polar"}
)
fig_pat.suptitle(
    "Beam patterns  (freq-averaged over 500 / 1k / 2k / 3.5kHz)",
    fontsize=12
)

for ax, (label, deg) in zip(axes_pat, directions.items()):
    steer = steering_delays(deg)
    gains = []
    for theta in sweep:
        td_test = steering_delays(theta)
        delta = td_test - steer          # delay difference at test angle
        g = 0.0
        for f_hz in f_eval_hz:
            f_norm = f_hz / fs           # cycles per sample
            phases = np.exp(1j * 2 * np.pi * f_norm * delta)
            g += np.abs(np.mean(phases))
        gains.append(g / len(f_eval_hz))

    gains = np.array(gains)
    gains /= gains.max()
    ax.plot(np.radians(sweep), gains, linewidth=1.2)
    ax.set_title(label.replace("_", "\n"), fontsize=9, pad=10)
    ax.set_theta_zero_location("N")      # 0° at top = front
    ax.set_theta_direction(-1)           # clockwise
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
    ax.axvline(np.radians(deg), color="red", linewidth=1, alpha=0.6,
               label="steer dir")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "beam_patterns.png"), dpi=150)
print("\nSaved: beam_patterns.png")

# ── apply to mixture.wav ──────────────────────────────────────────────────
beamed_signals = {}
for label, deg in directions.items():
    td = all_tdoas[label]
    beamed_signals[label] = beamform(mx, td)

# ── spectrograms: absolute ────────────────────────────────────────────────
fig_spec, axes_spec = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig_spec.suptitle(
    "Geometry-based freq-domain DAS  —  mixture.wav\n"
    "(τ_lr=29samp, τ_fr=8samp, fractional delays)",
    fontsize=12
)
for ax, (label, sig) in zip(axes_spec, beamed_signals.items()):
    ax.specgram(sig, Fs=fs, NFFT=1024, noverlap=512, cmap="inferno", scale="dB")
    ax.set_ylim(0, 8000)
    ax.set_ylabel(label.replace("_", "\n"), fontsize=9)
axes_spec[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spectrograms_abs.png"), dpi=150)
print("Saved: spectrograms_abs.png")

# ── spectrograms: each beam minus the mean across all beams ───────────────
# This highlights what is UNIQUE to each direction, suppressing
# content that is equally present in all beams (i.e. omnidirectional sound).
mean_sig = np.mean(list(beamed_signals.values()), axis=0)

fig_diff, axes_diff = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig_diff.suptitle(
    "Differential spectrograms  (beam − mean of all beams)\n"
    "Highlights content unique to each direction",
    fontsize=12
)
for ax, (label, sig) in zip(axes_diff, beamed_signals.items()):
    diff = sig - mean_sig
    ax.specgram(diff, Fs=fs, NFFT=1024, noverlap=512, cmap="RdBu_r", scale="dB")
    ax.set_ylim(0, 8000)
    ax.set_ylabel(label.replace("_", "\n"), fontsize=9)
axes_diff[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spectrograms_diff.png"), dpi=150)
print("Saved: spectrograms_diff.png")

# ── save WAVs ─────────────────────────────────────────────────────────────
for label, sig in beamed_signals.items():
    sig_norm = sig / (np.max(np.abs(sig)) + 1e-9)
    out_path = os.path.join(OUT_DIR, f"beam_{label}.wav")
    wav.write(out_path, fs, (sig_norm * np.iinfo(np.int16).max).astype(np.int16))
    print(f"Saved: beam_{label}.wav")

print("\nDone.")
