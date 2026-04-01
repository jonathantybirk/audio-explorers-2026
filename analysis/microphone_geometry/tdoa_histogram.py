"""
Short-time GCC-PHAT histogram for TDoA reliability scoring.

Rather than one GCC-PHAT over the full signal, we compute a peak lag per
short frame and aggregate into a histogram. A true TDoA shows up as a sharp,
tall mode. We quantify each mode's prominence (height above surrounding
baseline) as an objective reliability score.

This resolves the LF–LR vs RF–RR disagreement from estimate_spacing.py:
whichever has a more prominent histogram peak at its claimed lag is the
more trustworthy intra-ear estimate.

Plots saved to analysis/microphone_geometry/tdoa_histograms/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

SPEED_OF_SOUND = 343.0  # m/s
FRAME_S = 0.05          # 50 ms frames — long enough for GCC-PHAT, short enough to track
HOP_S   = 0.025         # 50 % overlap
MAX_LAG_S = 0.001       # search window: ±1 ms covers any head-sized array

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
HIST_DIR  = os.path.join(os.path.dirname(__file__), "tdoa_histograms")
os.makedirs(HIST_DIR, exist_ok=True)

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


def gcc_phat_frame(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def tdoa_histogram(data, ch_a, ch_b, sr, frame_samples, hop_samples, max_lag_s):
    """
    Compute per-frame peak lags and return (bin_centers_s, histogram_counts).
    Each frame contributes one peak lag (the argmax of GCC-PHAT).
    """
    sig_a = data[:, ch_a]
    sig_b = data[:, ch_b]
    n = len(sig_a)
    n_fft = 1 << (frame_samples - 1).bit_length()  # next power of 2 >= frame

    max_lag_samples = int(max_lag_s * sr)
    center = n_fft // 2

    peak_lags = []
    for start in range(0, n - frame_samples, hop_samples):
        xa = sig_a[start : start + frame_samples]
        ya = sig_b[start : start + frame_samples]
        gcc = gcc_phat_frame(xa, ya, n_fft)

        # Only look within the search window, excluding zero-lag
        lo = center - max_lag_samples
        hi = center + max_lag_samples
        window = gcc[lo:hi].copy()

        # Exclude zero-lag region (±min_lag = 2 samples)
        mid = max_lag_samples
        window[mid - 2 : mid + 3] = 0.0

        idx_local = np.argmax(window)
        lag_samples = idx_local - max_lag_samples  # relative to centre
        peak_lags.append(lag_samples / sr)

    peak_lags = np.array(peak_lags)

    # Histogram with ~sample-resolution bins
    bin_width_s = 1 / sr
    bins = np.arange(-max_lag_s - bin_width_s / 2,
                      max_lag_s + bin_width_s,
                      bin_width_s)
    counts, edges = np.histogram(peak_lags, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def peak_prominence(counts, idx):
    """
    Prominence of counts[idx]: height above the higher of the two surrounding
    minima (one on each side). Standard peak-prominence definition.
    """
    left_min  = counts[:idx].min()  if idx > 0            else 0
    right_min = counts[idx+1:].min() if idx < len(counts)-1 else 0
    base = max(left_min, right_min)
    return counts[idx] - base


def analyse_pair(data, ch_a, ch_b, sr, frame_samples, hop_samples, label):
    la = CHANNEL_LABEL[ch_a]
    lb = CHANNEL_LABEL[ch_b]
    centers, counts = tdoa_histogram(data, ch_a, ch_b, sr,
                                     frame_samples, hop_samples, MAX_LAG_S)

    # Find peaks in the histogram with minimum height
    peaks, props = find_peaks(counts, height=5, distance=3)

    # Score each peak by prominence
    prominences = [peak_prominence(counts, p) for p in peaks]

    # Best peak (highest prominence)
    if prominences:
        best_idx = int(np.argmax(prominences))
        best_peak = peaks[best_idx]
        best_lag  = centers[best_peak]
        best_prom = prominences[best_idx]
        spacing_cm = abs(best_lag) * SPEED_OF_SOUND * 100
    else:
        best_lag, best_prom, spacing_cm = 0.0, 0.0, 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(centers * 1e3, counts, width=(centers[1] - centers[0]) * 1e3,
           color="steelblue", alpha=0.8, label="frame count")
    for i, p in enumerate(peaks):
        color = "crimson" if i == best_idx else "gray"
        ax.axvline(centers[p] * 1e3, color=color, linestyle="--", linewidth=1.2,
                   label=f"peak {centers[p]*1e3:+.2f} ms  prom={prominences[i]:.0f}" if i == best_idx else None)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Frame count")
    ax.set_title(f"{la}–{lb}  |  {label}\n"
                 f"Best peak: τ = {best_lag*1e3:+.3f} ms  →  spacing = {spacing_cm:.2f} cm  "
                 f"(prominence = {best_prom:.0f} frames)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(HIST_DIR, f"hist_{la}_{lb}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  {la}–{lb}:  τ = {best_lag*1e3:+.3f} ms  →  {spacing_cm:.2f} cm"
          f"  (prominence = {best_prom:.0f} frames)  → {fname.split('/')[-1]}")
    return best_lag, best_prom, spacing_cm


# ── Load ──────────────────────────────────────────────────────────────────────
sr, data = load_wav(WAV_PATH)
print(f"Loaded example_mixture: sr={sr} Hz, shape={data.shape}")
frame_samples = int(FRAME_S * sr)
hop_samples   = int(HOP_S   * sr)
print(f"Frame: {frame_samples} samples ({FRAME_S*1000:.0f} ms), "
      f"hop: {hop_samples} samples ({HOP_S*1000:.0f} ms)\n")

# ── Inter-ear D ───────────────────────────────────────────────────────────────
print("── Inter-ear distance D ─────────────────────────────────────────────────")
tau_D1, prom_D1, D1 = analyse_pair(data, LF, RF, sr, frame_samples, hop_samples, "inter-ear primary")
tau_D2, prom_D2, D2 = analyse_pair(data, LR, RR, sr, frame_samples, hop_samples, "inter-ear cross-check")

D_winner = D1 if prom_D1 >= prom_D2 else D2
D_winner_label = "LF–RF" if prom_D1 >= prom_D2 else "LR–RR"
print(f"  → Trusting {D_winner_label} (higher prominence).  D = {D_winner:.2f} cm\n")

# ── Intra-ear L ───────────────────────────────────────────────────────────────
print("── Intra-ear spacing L ──────────────────────────────────────────────────")
tau_L1, prom_L1, L1 = analyse_pair(data, LF, LR, sr, frame_samples, hop_samples, "intra-ear left primary")
tau_L2, prom_L2, L2 = analyse_pair(data, RF, RR, sr, frame_samples, hop_samples, "intra-ear right cross-check")

L_winner = L1 if prom_L1 >= prom_L2 else L2
L_winner_label = "LF–LR" if prom_L1 >= prom_L2 else "RF–RR"
print(f"  → Trusting {L_winner_label} (higher prominence).  L = {L_winner:.2f} cm\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("════════════════════════════════════════════════════════════════════════")
print("  Statistically selected microphone geometry")
print(f"  Inter-ear D = {D_winner:.2f} cm  (from {D_winner_label}, "
      f"prominence {max(prom_D1,prom_D2):.0f} vs {min(prom_D1,prom_D2):.0f})")
print(f"  Intra-ear L = {L_winner:.2f} cm  (from {L_winner_label}, "
      f"prominence {max(prom_L1,prom_L2):.0f} vs {min(prom_L1,prom_L2):.0f})")
print("════════════════════════════════════════════════════════════════════════")
