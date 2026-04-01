"""
SRP-PHAT azimuth scan on example_mixture.wav.

Uses the geometry from data/mic_geometry.json to scan 360° and produce a
spatial power spectrum.  With a correct geometry we expect 4 peaks at
0°, 90°, 180°, 270°.  A clean result validates the mic geometry; blurry
or shifted peaks would indicate a geometry problem.

Plots saved to analysis/srp_phat/spatial_spectrum/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH   = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
GEO_PATH   = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
PLOT_DIR   = os.path.join(os.path.dirname(__file__), "spatial_spectrum")
os.makedirs(PLOT_DIR, exist_ok=True)

LF, LR, RF, RR = 0, 1, 2, 3

# ── Load geometry ─────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
print(f"Geometry: D={D*100:.2f} cm, L={L*100:.2f} cm")

MIC_POS = {
    LF: ( D/2,  L/2),
    LR: ( D/2, -L/2),
    RF: (-D/2,  L/2),
    RR: (-D/2, -L/2),
}


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


def predicted_tdoa(ch_a, ch_b, phi_rad):
    xa, ya = MIC_POS[ch_a]
    xb, yb = MIC_POS[ch_b]
    return ((xa - xb) * np.sin(phi_rad) + (ya - yb) * np.cos(phi_rad)) / C


# ── Load & compute GCC-PHAT for all 6 pairs ───────────────────────────────────
sr, data = load_wav(WAV_PATH)
n        = data.shape[0]
n_fft    = 1 << (n - 1).bit_length()
lags     = (np.arange(n_fft) - n_fft // 2) / sr

import itertools
all_pairs = list(itertools.combinations([LF, LR, RF, RR], 2))
gcc_store = {(a, b): gcc_phat(data[:, a], data[:, b], n_fft) for a, b in all_pairs}
print(f"GCC-PHAT computed for {len(all_pairs)} pairs")

# ── SRP-PHAT scan ─────────────────────────────────────────────────────────────
azimuths = np.linspace(0, 360, 720, endpoint=False)  # 0.5° resolution
power    = np.zeros(len(azimuths))

for k, az in enumerate(azimuths):
    phi = np.deg2rad(az)
    s = 0.0
    for ch_a, ch_b in all_pairs:
        tau = predicted_tdoa(ch_a, ch_b, phi)
        # linear interpolation into gcc array
        lag_s   = tau
        lag_idx = (lag_s * sr) + n_fft // 2
        lo      = int(np.floor(lag_idx))
        hi      = lo + 1
        frac    = lag_idx - lo
        lo      = np.clip(lo, 0, n_fft - 1)
        hi      = np.clip(hi, 0, n_fft - 1)
        s += (1 - frac) * gcc_store[(ch_a, ch_b)][lo] + frac * gcc_store[(ch_a, ch_b)][hi]
    power[k] = s

# ── Cartesian plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(azimuths, power, linewidth=0.9, color="steelblue")
for expected in [0, 90, 180, 270]:
    ax.axvline(expected, color="crimson", linestyle="--", alpha=0.6,
               label="expected" if expected == 0 else None)
ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("SRP-PHAT power")
ax.set_title("SRP-PHAT spatial spectrum — example_mixture.wav  (all 6 mic pairs)")
ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax.legend()
plt.tight_layout()
cart_path = os.path.join(PLOT_DIR, "srp_phat_cartesian.png")
plt.savefig(cart_path, dpi=150)
plt.close()
print(f"Saved: {cart_path}")

# ── Polar plot ────────────────────────────────────────────────────────────────
power_norm = (power - power.min()) / (power.max() - power.min())
phi_rad    = np.deg2rad(azimuths)

# Convention: 0=front=top, 90=left=left  →  plot angle = 90° - azimuth
plot_angle = np.deg2rad(90 - azimuths)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
ax.plot(plot_angle, power_norm, linewidth=1.2, color="steelblue")
ax.fill(plot_angle, power_norm, alpha=0.25, color="steelblue")

for expected_az, label in [(0, "front"), (90, "left"), (180, "back"), (270, "right")]:
    pa = np.deg2rad(90 - expected_az)
    ax.axvline(pa, color="crimson", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(pa, 1.12, label, ha="center", va="center", fontsize=9, color="crimson")

ax.set_theta_zero_location("N")   # 0° at top
ax.set_theta_direction(-1)        # clockwise
ax.set_rticks([])
ax.set_title("SRP-PHAT spatial spectrum\nexample_mixture.wav", pad=20)
plt.tight_layout()
polar_path = os.path.join(PLOT_DIR, "srp_phat_polar.png")
plt.savefig(polar_path, dpi=150)
plt.close()
print(f"Saved: {polar_path}")

# ── Find and report peaks ─────────────────────────────────────────────────────
from scipy.signal import find_peaks
peaks, _ = find_peaks(power, distance=len(azimuths) // 16, height=np.percentile(power, 80))
print(f"\nDetected peaks at: {[f'{azimuths[p]:.1f}°' for p in peaks]}")
print(f"Expected:          ['0.0°', '90.0°', '180.0°', '270.0°']")
