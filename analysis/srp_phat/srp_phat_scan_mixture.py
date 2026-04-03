"""
SRP-PHAT azimuth scan on mixture.wav.

Identical pipeline to srp_phat_scan.py but applied to mixture.wav where
the speaker positions are unknown.  The detected peaks become the estimated
steering angles used by downstream beamforming.

Detected angles are saved to analysis/srp_phat/mixture_angles.json for use
by demo/generate_audio_mixture.py and other scripts.

Outputs saved to analysis/srp_phat/mixture_spectrum/
"""

import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "mixture_spectrum")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_PEAKS          = 4      # expected number of sources
MIN_SEPARATION   = 40.0   # minimum angular separation between peaks (degrees)
PEAK_PERCENTILE  = 75     # minimum height threshold (percentile of power)

LF, LR, RF, RR = 0, 1, 2, 3


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    return sr, data.astype(np.float64)


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


# ── Load geometry ─────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
print(f"Geometry: D={D*100:.2f} cm, L={L*100:.2f} cm, c={C:.1f} m/s")

MIC_POS = {
    LF: ( D/2,  L/2),
    LR: ( D/2, -L/2),
    RF: (-D/2,  L/2),
    RR: (-D/2, -L/2),
}


def predicted_tdoa(ch_a, ch_b, phi_rad):
    xa, ya = MIC_POS[ch_a]
    xb, yb = MIC_POS[ch_b]
    return ((xa - xb) * np.sin(phi_rad) + (ya - yb) * np.cos(phi_rad)) / C


# ── Load & compute GCC-PHAT for all 6 pairs ───────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
      f"|  {sr} Hz  |  {data.shape[0]/sr:.1f} s")

n     = data.shape[0]
n_fft = 1 << (n - 1).bit_length()
lags  = (np.arange(n_fft) - n_fft // 2) / sr

all_pairs = list(itertools.combinations([LF, LR, RF, RR], 2))
gcc_store = {(a, b): gcc_phat(data[:, a], data[:, b], n_fft) for a, b in all_pairs}
print(f"GCC-PHAT computed for {len(all_pairs)} pairs")

# ── SRP-PHAT scan ─────────────────────────────────────────────────────────────
azimuths = np.linspace(0, 360, 720, endpoint=False)
power    = np.zeros(len(azimuths))

for k, az in enumerate(azimuths):
    phi = np.deg2rad(az)
    s   = 0.0
    for ch_a, ch_b in all_pairs:
        tau     = predicted_tdoa(ch_a, ch_b, phi)
        lag_idx = (tau * sr) + n_fft // 2
        lo      = int(np.floor(lag_idx)); hi = lo + 1
        frac    = lag_idx - lo
        lo      = np.clip(lo, 0, n_fft - 1); hi = np.clip(hi, 0, n_fft - 1)
        s += (1 - frac) * gcc_store[(ch_a, ch_b)][lo] + frac * gcc_store[(ch_a, ch_b)][hi]
    power[k] = s

# ── Peak detection ────────────────────────────────────────────────────────────
min_dist_bins = max(1, int(MIN_SEPARATION / (360 / len(azimuths))))
peaks, props  = find_peaks(power, distance=min_dist_bins,
                           height=np.percentile(power, PEAK_PERCENTILE))

# Take the N_PEAKS strongest
if len(peaks) >= N_PEAKS:
    top_idx = np.argsort(props["peak_heights"])[-N_PEAKS:]
    peaks   = np.sort(peaks[top_idx])
detected_angles = [float(azimuths[p]) for p in peaks]

print(f"\nDetected {len(detected_angles)} peaks:")
for az in detected_angles:
    print(f"  {az:.1f}°")

# ── Save detected angles for downstream scripts ───────────────────────────────
angles_path = os.path.join(os.path.dirname(__file__), "mixture_angles.json")
angles_data = {
    "wav": os.path.basename(WAV_PATH),
    "method": "SRP-PHAT all 6 mic pairs",
    "estimated_angles_deg": detected_angles,
    "azimuth_convention": geo["azimuth_convention"],
}
with open(angles_path, "w") as f:
    json.dump(angles_data, f, indent=2)
print(f"\nSaved estimated angles: {angles_path}")

# ── Cartesian plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(azimuths, power, linewidth=0.9, color="#7c3aed")
for az in detected_angles:
    ax.axvline(az, color="#0f766e", linestyle="--", alpha=0.8, linewidth=1.2)
    ax.text(az + 2, ax.get_ylim()[1] * 0.95, f"{az:.0f}°", color="#0f766e",
            fontsize=9, va="top")
ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("SRP-PHAT power")
ax.set_title("SRP-PHAT spatial spectrum — mixture.wav  (all 6 mic pairs)")
ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
plt.tight_layout()
cart_path = os.path.join(PLOT_DIR, "srp_phat_mixture_cartesian.png")
plt.savefig(cart_path, dpi=150)
plt.close()
print(f"Saved: {cart_path}")

# ── Polar plot ────────────────────────────────────────────────────────────────
power_norm = (power - power.min()) / (power.max() - power.min())
az_rad     = np.deg2rad(azimuths)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(1)
ax.plot(az_rad, power_norm, linewidth=1.2, color="steelblue")
ax.fill(az_rad, power_norm, alpha=0.25, color="steelblue")
for az in detected_angles:
    pa = np.deg2rad(az)
    ax.axvline(pa, color="crimson", linestyle="--", linewidth=1, alpha=0.7)
for p in peaks:
    pa = np.deg2rad(azimuths[p])
    ax.plot([pa], [power_norm[p]], marker="o", markersize=5, color="#0f766e")
ax.set_thetagrids([0, 90, 180, 270],
                  labels=["Front\n0°", "Left\n90°", "Back\n180°", "Right\n270°"],
                  fontsize=10, color="#c0392b")
ax.set_rticks([])
ax.set_title("SRP-PHAT spatial spectrum — mixture.wav", pad=24)
plt.tight_layout()
polar_path = os.path.join(PLOT_DIR, "srp_phat_mixture_polar.png")
plt.savefig(polar_path, dpi=150)
plt.close()
print(f"Saved: {polar_path}")
