"""
Estimate microphone spacings from example_mixture.wav using GCC-PHAT.

Physical model (far field, azimuth φ from front, 90°=left):

    τ(LF, RF, φ) = D · sin(φ) / c    →  maximised at φ=90° (left talker)
    τ(LF, LR, φ) = L · cos(φ) / c    →  maximised at φ=0°  (front talker)

Since we know the four talkers in example_mixture sit at exactly 0°/90°/180°/270°,
the outermost GCC-PHAT peaks directly give D and L without any assumption about
head dimensions.

Each spacing is estimated from two independent mic pairs and the results are compared
as a sanity check:
  Inter-ear D:  LF–RF  and  LR–RR   (both span the full left–right axis)
  Intra-ear L:  LF–LR  and  RF–RR   (both span the front–rear axis on each ear)

Plots saved to analysis/microphone_geometry/gcc_plots/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

SPEED_OF_SOUND = 343.0  # m/s

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "gcc_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Channel indices as per case spec: [LF, LR, RF, RR]
LF, LR, RF, RR = 0, 1, 2, 3
CHANNEL_LABEL = {LF: "LF", LR: "LR", RF: "RF", RR: "RR"}


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def gcc_phat(x, y, sr):
    """
    Generalised Cross-Correlation with Phase Transform.
    Returns (lags_in_seconds, gcc_values).
    """
    n_fft = 1 << (len(x) + len(y) - 2).bit_length()
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    gcc = np.fft.fftshift(np.fft.irfft(G, n=n_fft))
    lags = (np.arange(n_fft) - n_fft // 2) / sr
    return lags, gcc


def peak_lag(lags, gcc, max_lag_s, min_lag_s=2e-4):
    """
    Largest peak within ±max_lag_s, skipping the zero-lag region.

    With 4 simultaneous sources at 0°/90°/180°/270° the zero-lag peak is
    inflated by two sources at once (e.g. 0° and 180° for LF–RF).
    Skipping |lag| < min_lag_s lets us find the ±τ peaks we actually want.
    """
    mask = (np.abs(lags) >= min_lag_s) & (np.abs(lags) <= max_lag_s)
    idx = np.argmax(gcc[mask])
    return lags[mask][idx]


def plot_gcc(lags, gcc, ch_a, ch_b, axis_label, peak, max_lag_s):
    label_a = CHANNEL_LABEL[ch_a]
    label_b = CHANNEL_LABEL[ch_b]
    mask = np.abs(lags) <= max_lag_s

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(lags[mask] * 1e3, gcc[mask], linewidth=0.8, color="steelblue")
    ax.axvline( peak * 1e3, color="crimson",  linestyle="--",
                label=f"peak  {peak*1e3:+.3f} ms  →  {abs(peak)*SPEED_OF_SOUND*100:.1f} cm")
    ax.axvline(-peak * 1e3, color="darkorange", linestyle="--",
                label=f"mirror {-peak*1e3:+.3f} ms")
    ax.axvspan(-2e-4 * 1e3, 2e-4 * 1e3, alpha=0.12, color="gray",
               label="excluded zero-lag region")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("GCC-PHAT")
    ax.set_title(f"{label_a} vs {label_b}  —  {axis_label}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"gcc_{label_a}_{label_b}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ── Load ──────────────────────────────────────────────────────────────────────
sr, data = load_wav(WAV_PATH)
print(f"Loaded example_mixture: sr={sr} Hz, shape={data.shape}\n")

MAX_LAG_S = 0.001  # 1 ms ≈ 34 cm — safely covers any realistic head


# ── Inter-ear distance D ──────────────────────────────────────────────────────
# τ(A, B) = D·sin(φ)/c  for any left–right mic pair
# Peak at φ=90° (left talker) → τ_max = D/c

print("── Inter-ear distance D ─────────────────────────────────────────────────")

lags, gcc = gcc_phat(data[:, LF], data[:, RF], sr)
tau1 = peak_lag(lags, gcc, MAX_LAG_S)
D1 = abs(tau1) * SPEED_OF_SOUND
plot_gcc(lags, gcc, LF, RF, "inter-ear  (primary)", tau1, MAX_LAG_S)
print(f"  LF–RF:  τ = {tau1*1e3:+.3f} ms  →  D = {D1*100:.2f} cm")

lags, gcc = gcc_phat(data[:, LR], data[:, RR], sr)
tau2 = peak_lag(lags, gcc, MAX_LAG_S)
D2 = abs(tau2) * SPEED_OF_SOUND
plot_gcc(lags, gcc, LR, RR, "inter-ear  (cross-check)", tau2, MAX_LAG_S)
print(f"  LR–RR:  τ = {tau2*1e3:+.3f} ms  →  D = {D2*100:.2f} cm")

D = (D1 + D2) / 2
print(f"  Agreement: {abs(D1 - D2)*100:.2f} cm difference  |  mean D = {D*100:.2f} cm")


# ── Intra-ear spacing L ───────────────────────────────────────────────────────
# τ(A, B) = L·cos(φ)/c  for any front–rear mic pair on the same ear
# Peak at φ=0° (front talker) → τ_max = L/c

print("\n── Intra-ear spacing L ──────────────────────────────────────────────────")

lags, gcc = gcc_phat(data[:, LF], data[:, LR], sr)
tau3 = peak_lag(lags, gcc, MAX_LAG_S)
L1 = abs(tau3) * SPEED_OF_SOUND
plot_gcc(lags, gcc, LF, LR, "intra-ear left  (primary)", tau3, MAX_LAG_S)
print(f"  LF–LR:  τ = {tau3*1e3:+.3f} ms  →  L = {L1*100:.2f} cm")

lags, gcc = gcc_phat(data[:, RF], data[:, RR], sr)
tau4 = peak_lag(lags, gcc, MAX_LAG_S)
L2 = abs(tau4) * SPEED_OF_SOUND
plot_gcc(lags, gcc, RF, RR, "intra-ear right  (cross-check)", tau4, MAX_LAG_S)
print(f"  RF–RR:  τ = {tau4*1e3:+.3f} ms  →  L = {L2*100:.2f} cm")

L = (L1 + L2) / 2
print(f"  Agreement: {abs(L1 - L2)*100:.2f} cm difference  |  mean L = {L*100:.2f} cm")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n════════════════════════════════════════════════════════════════════════")
print("  Microphone geometry (estimated from signal, no head-size assumption)")
print(f"  Inter-ear distance  D = {D*100:.2f} cm  (LF–RF: {D1*100:.2f}, LR–RR: {D2*100:.2f})")
print(f"  Intra-ear spacing   L = {L*100:.2f} cm  (LF–LR: {L1*100:.2f}, RF–RR: {L2*100:.2f})")
print(f"  Speed of sound assumed: {SPEED_OF_SOUND} m/s")
print("  These are fixed hardware properties — identical for mixture.wav.")
print("════════════════════════════════════════════════════════════════════════")
