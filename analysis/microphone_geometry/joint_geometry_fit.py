"""
Joint geometry estimation via SRP-style grid search over (D, L).

Instead of estimating inter-ear distance D and intra-ear spacing L from
individual mic pairs in isolation, this script uses ALL 6 pairwise GCC-PHAT
functions simultaneously.

For a candidate geometry (D, L) we predict the expected TDoA for every mic
pair at every known source direction (0°/90°/180°/270°), look up the
GCC-PHAT value at each predicted lag, and sum them. The (D, L) that
maximises this sum is the geometry most consistent with all available data.
This is the SRP-PHAT (Steered Response Power) principle applied to geometry
calibration rather than DoA estimation.

Mic layout assumed (x = left-positive, y = front-positive):
    LF: (+D/2, +L/2)    LR: (+D/2, -L/2)
    RF: (-D/2, +L/2)    RR: (-D/2, -L/2)

Predicted TDoA for pair (i, j) at azimuth φ (0° = front, 90° = left):
    τ_ij(φ) = [(x_i - x_j)·sin(φ) + (y_i - y_j)·cos(φ)] / c

Plots saved to analysis/microphone_geometry/joint_fit_plots/
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

SPEED_OF_SOUND = 343.0  # m/s

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "joint_fit_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Channel indices: [LF, LR, RF, RR]
LF, LR, RF, RR = 0, 1, 2, 3
CHANNEL_LABEL  = {LF: "LF", LR: "LR", RF: "RF", RR: "RR"}

# Known source azimuths in the calibration recording (radians, 0=front, 90=left)
SOURCE_AZIMUTHS_DEG = [0.0, 90.0, 180.0, 270.0]
SOURCE_AZIMUTHS_RAD = [np.deg2rad(a) for a in SOURCE_AZIMUTHS_DEG]

# Grid search ranges
D_RANGE = np.linspace(0.10, 0.35, 200)   # inter-ear distance, metres
L_RANGE = np.linspace(0.01, 0.18, 200)   # intra-ear spacing, metres


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def mic_positions(D, L):
    """Return dict of mic positions {ch: (x, y)} for given D and L."""
    return {
        LF: ( D/2,  L/2),
        LR: ( D/2, -L/2),
        RF: (-D/2,  L/2),
        RR: (-D/2, -L/2),
    }


def predicted_tdoa(pos, ch_a, ch_b, phi_rad):
    """Far-field TDoA for pair (ch_a, ch_b) at azimuth phi (radians)."""
    xa, ya = pos[ch_a]
    xb, yb = pos[ch_b]
    return ((xa - xb) * np.sin(phi_rad) + (ya - yb) * np.cos(phi_rad)) / SPEED_OF_SOUND


def gcc_at_lag(gcc, lags, tau):
    """Interpolate GCC-PHAT value at lag tau (seconds)."""
    idx = np.searchsorted(lags, tau)
    idx = np.clip(idx, 0, len(gcc) - 1)
    return gcc[idx]


# ── Load & compute all 6 GCC-PHAT functions ───────────────────────────────────
sr, data = load_wav(WAV_PATH)
print(f"Loaded example_mixture: sr={sr} Hz, shape={data.shape}")

n_fft = 1 << (len(data[:, 0]) - 1).bit_length()
lags  = (np.arange(n_fft) - n_fft // 2) / sr

all_pairs = list(itertools.combinations([LF, LR, RF, RR], 2))
gcc_store = {}
for ch_a, ch_b in all_pairs:
    gcc_store[(ch_a, ch_b)] = gcc_phat(data[:, ch_a], data[:, ch_b], n_fft)
    print(f"  GCC-PHAT computed: {CHANNEL_LABEL[ch_a]}–{CHANNEL_LABEL[ch_b]}")

# ── Grid search ───────────────────────────────────────────────────────────────
print("\nRunning grid search over (D, L)...")
score_grid = np.zeros((len(D_RANGE), len(L_RANGE)))

for i, D in enumerate(D_RANGE):
    for j, L in enumerate(L_RANGE):
        pos = mic_positions(D, L)
        score = 0.0
        for ch_a, ch_b in all_pairs:
            gcc = gcc_store[(ch_a, ch_b)]
            for phi in SOURCE_AZIMUTHS_RAD:
                tau = predicted_tdoa(pos, ch_a, ch_b, phi)
                score += gcc_at_lag(gcc, lags, tau)
        score_grid[i, j] = score

# ── Find best (D, L) ──────────────────────────────────────────────────────────
best_i, best_j = np.unravel_index(np.argmax(score_grid), score_grid.shape)
D_best = D_RANGE[best_i]
L_best = L_RANGE[best_j]

print(f"\nBest fit:")
print(f"  Inter-ear distance  D = {D_best*100:.2f} cm")
print(f"  Intra-ear spacing   L = {L_best*100:.2f} cm")

# ── Plot: 2-D score landscape ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
img = ax.imshow(
    score_grid.T,
    origin="lower",
    aspect="auto",
    extent=[D_RANGE[0]*100, D_RANGE[-1]*100, L_RANGE[0]*100, L_RANGE[-1]*100],
    cmap="viridis",
)
ax.plot(D_best*100, L_best*100, "r*", markersize=14, label=f"best: D={D_best*100:.1f} cm, L={L_best*100:.1f} cm")
ax.set_xlabel("Inter-ear distance D (cm)")
ax.set_ylabel("Intra-ear spacing L (cm)")
ax.set_title("SRP-style geometry score — all 6 pairs × 4 known source directions")
ax.legend()
fig.colorbar(img, ax=ax, label="Summed GCC-PHAT response")
plt.tight_layout()
landscape_path = os.path.join(PLOT_DIR, "geometry_score_landscape.png")
plt.savefig(landscape_path, dpi=150)
plt.close()
print(f"\nSaved: {landscape_path}")

# ── Plot: 1-D slices through the optimum ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(D_RANGE * 100, score_grid[:, best_j])
ax1.axvline(D_best * 100, color="red", linestyle="--", label=f"D = {D_best*100:.2f} cm")
ax1.set_xlabel("Inter-ear distance D (cm)")
ax1.set_ylabel("Score (L fixed at optimum)")
ax1.set_title("D slice")
ax1.legend()

ax2.plot(L_RANGE * 100, score_grid[best_i, :])
ax2.axvline(L_best * 100, color="red", linestyle="--", label=f"L = {L_best*100:.2f} cm")
ax2.set_xlabel("Intra-ear spacing L (cm)")
ax2.set_ylabel("Score (D fixed at optimum)")
ax2.set_title("L slice")
ax2.legend()

plt.suptitle("Score profiles through best-fit geometry", y=1.02)
plt.tight_layout()
slices_path = os.path.join(PLOT_DIR, "geometry_score_slices.png")
plt.savefig(slices_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {slices_path}")

# ── Save geometry for downstream scripts ─────────────────────────────────────
import json

GEOMETRY_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
os.makedirs(os.path.dirname(GEOMETRY_PATH), exist_ok=True)
geometry = {
    "inter_ear_distance_m": round(float(D_best), 6),
    "intra_ear_spacing_m":  round(float(L_best), 6),
    "speed_of_sound_m_s":   SPEED_OF_SOUND,
    "method": "joint SRP-style grid search, all 6 mic pairs x 4 known source directions",
    "channel_order": ["LF", "LR", "RF", "RR"],
    "azimuth_convention": "degrees, 0=front, 90=left, 180=back, 270=right",
}
with open(GEOMETRY_PATH, "w") as f:
    json.dump(geometry, f, indent=2)
print(f"\nGeometry saved: {GEOMETRY_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n════════════════════════════════════════════════════════════════════════")
print("  Joint geometry fit  (all 6 pairs × 4 known source directions)")
print(f"  Inter-ear distance  D = {D_best*100:.2f} cm")
print(f"  Intra-ear spacing   L = {L_best*100:.2f} cm")
print(f"  Speed of sound:       {SPEED_OF_SOUND} m/s")
print("  Applies equally to mixture.wav (same hardware).")
print("════════════════════════════════════════════════════════════════════════")
