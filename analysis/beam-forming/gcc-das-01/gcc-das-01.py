"""
gcc-das-01  —  GCC-PHAT Calibrated Delay-and-Sum Beamforming
==============================================================

Steering delays are measured entirely from example_mixture.wav
(4 known sources: 0°, 90°, 180°, 270°).  No microphone geometry
or physical distances are assumed anywhere.

Channel order: [LF=0, LR=1, RF=2, RR=3]

How we read each TDOA (relative to LF):
  τ_RF  per direction  ←  GCC-PHAT(LF, RF)
        Left/right-sensitive pair: 0° and 180° sources land near zero lag,
        90° and 270° sources give the two clear off-centre peaks.

  τ_LR  per direction  ←  GCC-PHAT(LF, LR)
        Front/rear-sensitive pair: 90° and 270° sources land near zero lag,
        0° and 180° sources give the two clear off-centre peaks.

  τ_RR  per direction  ←  GCC-PHAT(LF, RR)
        Diagonal pair (rear + right): all 4 sources give distinct peaks.
        We identify which peak belongs to which direction by seeding a
        small search window around the τ_LR and τ_RF values measured above
        (RR shares its rear behaviour with LR, and its right behaviour
        with RF — inferred from the channel labels, not from geometry).

Sign convention: positive τ  →  mic arrives τ samples AFTER LF.
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE    = "/Users/jonathantybirk/Desktop/Audio Explorers 2026"
EXAMPLE = os.path.join(BASE, "data/example_mixture.wav")
MIXTURE = os.path.join(BASE, "data/mixture.wav")
OUT_DIR = os.path.join(BASE, "analysis/beam-forming/gcc-das-01")

CH_LF, CH_LR, CH_RF, CH_RR = 0, 1, 2, 3

# ── load ───────────────────────────────────────────────────────────────────
fs, ex = wav.read(EXAMPLE)
_,  mx = wav.read(MIXTURE)
ex = ex.astype(np.float64) / np.iinfo(ex.dtype).max
mx = mx.astype(np.float64) / np.iinfo(mx.dtype).max

print(f"Sample rate : {fs} Hz")
print(f"Example     : {ex.shape[0]/fs:.2f}s   shape={ex.shape}")
print(f"Mixture     : {mx.shape[0]/fs:.2f}s   shape={mx.shape}")

# ── GCC-PHAT ───────────────────────────────────────────────────────────────
def gcc_phat(x, y, max_delay_ms=2.0):
    """
    Returns (r, lags).
    Peak at lag τ > 0  →  x leads y (y arrives τ samples after x).
    Peak at lag τ < 0  →  y leads x (x arrives |τ| samples after y).
    """
    n_fft = 2 ** int(np.ceil(np.log2(2 * len(x))))
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-10
    r = np.fft.irfft(R, n=n_fft)
    r = np.concatenate([r[n_fft // 2:], r[:n_fft // 2]])
    lags = np.arange(-n_fft // 2, n_fft // 2)
    cap = int(max_delay_ms * fs / 1000)
    mask = np.abs(lags) <= cap
    return r[mask], lags[mask]

def peak_on_side(r, lags, side, exclude_ms=0.05):
    """
    Find the argmax on the positive or negative side of the lag axis,
    excluding the central region within exclude_ms of zero.
    """
    excl = max(1, int(exclude_ms * fs / 1000))
    r = r.copy()
    r[np.abs(lags) <= excl] = 0
    r[lags < 0] = 0 if side == 'pos' else r[lags < 0]
    r[lags > 0] = 0 if side == 'neg' else r[lags > 0]
    return int(lags[np.argmax(r)])

def peak_near(r, lags, target, window_ms=0.25):
    """Find the argmax within ±window_ms of target lag."""
    w = max(1, int(window_ms * fs / 1000))
    mask = (lags >= target - w) & (lags <= target + w)
    return int(lags[mask][np.argmax(r[mask])])

# ── Step 1: measure all steering delays from example_mixture ──────────────
r_lf_rf, lags_lf_rf = gcc_phat(ex[:, CH_LF], ex[:, CH_RF])
r_lf_lr, lags_lf_lr = gcc_phat(ex[:, CH_LF], ex[:, CH_LR])
r_lf_rr, lags_lf_rr = gcc_phat(ex[:, CH_LF], ex[:, CH_RR])

# GCC(LF, RF): left-right sensitive pair
#   left  source (90°)  → LF leads RF  → positive peak  (LF arrives first)
#   right source (270°) → RF leads LF  → negative peak  (RF arrives first)
tau_rf_90  = peak_on_side(r_lf_rf, lags_lf_rf, 'pos')
tau_rf_270 = peak_on_side(r_lf_rf, lags_lf_rf, 'neg')

# GCC(LF, LR): front-rear sensitive pair
#   front source (0°)   → LF leads LR  → positive peak  (LF is in front)
#   back  source (180°) → LR leads LF  → negative peak  (LR is in back = closer)
tau_lr_0   = peak_on_side(r_lf_lr, lags_lf_lr, 'pos')
tau_lr_180 = peak_on_side(r_lf_lr, lags_lf_lr, 'neg')

# GCC(LF, RR): diagonal pair — all 4 sources give distinct peaks.
# RR is both rear and right, so:
#   front/back sources  behave like LR  → peaks near tau_lr_0 / tau_lr_180
#   left/right  sources behave like RF  → peaks near tau_rf_90 / tau_rf_270
tau_rr_0   = peak_near(r_lf_rr, lags_lf_rr, tau_lr_0)
tau_rr_180 = peak_near(r_lf_rr, lags_lf_rr, tau_lr_180)
tau_rr_90  = peak_near(r_lf_rr, lags_lf_rr, tau_rf_90)
tau_rr_270 = peak_near(r_lf_rr, lags_lf_rr, tau_rf_270)

# For the "near-zero" TDOAs on the symmetric pairs, we measure them too
# rather than hard-coding zero, so nothing is assumed.
tau_rf_0   = peak_near(r_lf_rf, lags_lf_rf, 0, window_ms=0.15)
tau_rf_180 = peak_near(r_lf_rf, lags_lf_rf, 0, window_ms=0.15)
tau_lr_90  = peak_near(r_lf_lr, lags_lf_lr, 0, window_ms=0.15)
tau_lr_270 = peak_near(r_lf_lr, lags_lf_lr, 0, window_ms=0.15)

# Full steering vectors: [τ_LF, τ_LR, τ_RF, τ_RR], all relative to LF (=0)
steering = {
    "0deg_front":  [0, tau_lr_0,   tau_rf_0,   tau_rr_0  ],
    "90deg_left":  [0, tau_lr_90,  tau_rf_90,  tau_rr_90 ],
    "180deg_back": [0, tau_lr_180, tau_rf_180, tau_rr_180],
    "270deg_right": [0, tau_lr_270, tau_rf_270, tau_rr_270],
}

def to_ms(s): return s / fs * 1000

print("\n── Empirical steering delays ──────────────────────────────────────────")
print(f"{'Direction':<14} {'τ_LF':>10} {'τ_LR':>10} {'τ_RF':>10} {'τ_RR':>10}")
print(f"{'':14} {'samp / ms':>10} {'samp / ms':>10} {'samp / ms':>10} {'samp / ms':>10}")
print("─" * 56)
for name, (tlf, tlr, trf, trr) in steering.items():
    print(f"{name:<14}  "
          f"{tlf:+3d}/{to_ms(tlf):+.2f}ms  "
          f"{tlr:+3d}/{to_ms(tlr):+.2f}ms  "
          f"{trf:+3d}/{to_ms(trf):+.2f}ms  "
          f"{trr:+3d}/{to_ms(trr):+.2f}ms")

# ── Step 2: plot GCC-PHAT curves with annotated peaks ─────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    "GCC-PHAT on example_mixture  —  empirical TDOA calibration\n"
    "(sources at 0°, 90°, 180°, 270°  ·  same mic setup as mixture.wav)",
    fontsize=12
)

colours = {"0deg_front": "tab:blue", "90deg_left": "tab:orange",
           "180deg_back": "tab:green", "270deg_right": "tab:red"}

panel_data = [
    ("GCC-PHAT(LF, RF)  — left/right sensitive",
     r_lf_rf, lags_lf_rf,
     {"90deg_left": tau_rf_90, "270deg_right": tau_rf_270,
      "0deg_front": tau_rf_0, "180deg_back": tau_rf_180}),
    ("GCC-PHAT(LF, LR)  — front/rear sensitive",
     r_lf_lr, lags_lf_lr,
     {"0deg_front": tau_lr_0, "180deg_back": tau_lr_180,
      "90deg_left": tau_lr_90, "270deg_right": tau_lr_270}),
    ("GCC-PHAT(LF, RR)  — diagonal (rear + right)",
     r_lf_rr, lags_lf_rr,
     {"0deg_front": tau_rr_0, "90deg_left": tau_rr_90,
      "180deg_back": tau_rr_180, "270deg_right": tau_rr_270}),
]

for ax, (title, r, lags, peaks) in zip(axes, panel_data):
    lags_ms = lags / fs * 1000
    ax.plot(lags_ms, r, color="steelblue", linewidth=0.7, zorder=1)
    for direction, tau in peaks.items():
        ax.axvline(tau / fs * 1000, color=colours[direction],
                   linestyle="--", linewidth=1.4,
                   label=f"{direction}  τ={tau:+d}samp ({to_ms(tau):+.2f}ms)")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("GCC-PHAT")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.25)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)

axes[-1].set_xlabel("Lag (ms)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gcc_phat_calibration.png"), dpi=150)
print("\nSaved: gcc_phat_calibration.png")

# ── Step 3: delay-and-sum beamforming ──────────────────────────────────────
def delay_and_sum(signals, tdoas):
    """
    Align each channel to the LF arrival time and average.
    tdoas[i] > 0  →  mic i is late  →  advance signal (shift array left)
    tdoas[i] < 0  →  mic i is early →  delay  signal (shift array right)
    """
    n = signals.shape[0]
    out = np.zeros(n)
    for i, tau in enumerate(tdoas):
        tau = int(round(tau))
        s = np.zeros(n)
        if tau > 0:
            s[:n - tau] = signals[tau:, i]
        elif tau < 0:
            s[-tau:] = signals[:n + tau, i]
        else:
            s = signals[:, i].copy()
        out += s
    return out / signals.shape[1]

# ── Step 4: apply to mixture.wav, save WAVs + spectrograms ────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(
    "Delay-and-Sum Beamforming on mixture.wav\n"
    "(steering delays calibrated from example_mixture.wav)",
    fontsize=12
)

for ax, (label, tdoa) in zip(axes, steering.items()):
    beamed = delay_and_sum(mx, tdoa)

    out_wav = os.path.join(OUT_DIR, f"beam_{label}.wav")
    wav.write(out_wav, fs, (beamed * np.iinfo(np.int16).max).astype(np.int16))

    ax.specgram(beamed, Fs=fs, NFFT=1024, noverlap=512, cmap="inferno", scale="dB")
    ax.set_ylim(0, 8000)
    ax.set_ylabel(label.replace("_", "\n"), fontsize=9)
    print(f"Saved: beam_{label}.wav")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "beamformed_spectrograms.png"), dpi=150)
print("Saved: beamformed_spectrograms.png")
print("\nDone.")
