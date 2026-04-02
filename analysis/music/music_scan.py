"""
Wideband MUSIC azimuth scan on example_mixture.wav.

This script uses short-block wideband MUSIC and aggregates the normalized
pseudo-spectrum over time.

Why block aggregation matters here:
  - Classical MUSIC needs fewer active sources than microphones.
  - We have 4 microphones and a 4-talker scene.
  - A full-recording, num_src=4 MUSIC run has no noise subspace and is
    therefore not a meaningful setup.

The workaround used here is practical rather than theoretical perfection:
  - run MUSIC on short STFT blocks
  - assume at most 3 dominant sources per block
  - aggregate the per-block spectra over the full recording

Outputs saved to analysis/music/spatial_spectrum/
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import stft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "spatial_spectrum")
os.makedirs(PLOT_DIR, exist_ok=True)

NFFT = 2048
HOP = 512
BLOCK_FRAMES = 24
BLOCK_HOP_FRAMES = BLOCK_FRAMES // 2
ACTIVE_SOURCES_PER_BLOCK = 3
FREQ_RANGE_HZ = [500.0, 3500.0]
AZIMUTHS_DEG = np.linspace(0, 360, 720, endpoint=False)  # 0.5° resolution


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def stack_stft(data, sr, nfft, hop):
    X = []
    for ch in range(data.shape[1]):
        _, _, Z = stft(
            data[:, ch],
            fs=sr,
            window="hann",
            nperseg=nfft,
            noverlap=nfft - hop,
            boundary=None,
            padded=False,
        )
        X.append(Z)
    return np.stack(X, axis=0)  # (M, F, T)


def circular_distance_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def select_distinct_peaks(power, azimuths_deg, num_peaks, min_separation_deg, min_percentile):
    prev_vals = np.roll(power, 1)
    next_vals = np.roll(power, -1)
    candidate_idx = np.where((power > prev_vals) & (power >= next_vals))[0]

    if len(candidate_idx) == 0:
        candidate_idx = np.arange(len(power))

    threshold = np.percentile(power, min_percentile)
    candidate_idx = candidate_idx[power[candidate_idx] >= threshold]
    if len(candidate_idx) == 0:
        candidate_idx = np.argsort(power)[-num_peaks:]

    ordered = candidate_idx[np.argsort(power[candidate_idx])[::-1]]
    chosen = []
    for idx in ordered:
        angle = azimuths_deg[idx]
        if all(circular_distance_deg(angle, azimuths_deg[j]) >= min_separation_deg for j in chosen):
            chosen.append(idx)
        if len(chosen) == num_peaks:
            break

    if len(chosen) < num_peaks:
        for idx in np.argsort(power)[::-1]:
            angle = azimuths_deg[idx]
            if all(circular_distance_deg(angle, azimuths_deg[j]) >= min_separation_deg for j in chosen):
                chosen.append(idx)
            if len(chosen) == num_peaks:
                break

    return np.array(sorted(chosen, key=lambda idx: azimuths_deg[idx]))


def nearest_cardinal(angle_deg):
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda ref: circular_distance_deg(angle_deg, ref))


# ── Load geometry ─────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)

D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
print(f"Geometry: D={D*100:.2f} cm, L={L*100:.2f} cm, c={C:.1f} m/s")

# pyroomacoustics defines azimuth with respect to the x-axis.
# We choose x=front-positive and y=left-positive so the repo's convention
# maps directly to MUSIC azimuths:
#   0°=front, 90°=left, 180°=back, 270°=right
MIC_POS = np.array([
    [L / 2, -L / 2, L / 2, -L / 2],   # x = front/back
    [D / 2, D / 2, -D / 2, -D / 2],   # y = left/right
], dtype=np.float64)


# ── Load mixture and compute STFT ─────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s"
)

print(
    "\nRunning block-aggregated MUSIC "
    f"(nfft={NFFT}, hop={HOP}, block_frames={BLOCK_FRAMES}, "
    f"active_sources={ACTIVE_SOURCES_PER_BLOCK}) ..."
)
X = stack_stft(data, sr, NFFT, HOP)
print(f"  STFT shape: {X.shape[0]} mics × {X.shape[1]} bins × {X.shape[2]} frames")

doa = pra.doa.MUSIC(
    MIC_POS,
    fs=sr,
    nfft=NFFT,
    c=C,
    num_src=ACTIVE_SOURCES_PER_BLOCK,
    azimuth=np.deg2rad(AZIMUTHS_DEG),
    mode="far",
    frequency_normalization=True,
)

power = np.zeros(len(AZIMUTHS_DEG), dtype=np.float64)
num_blocks = 0

for start in range(0, X.shape[2] - BLOCK_FRAMES + 1, BLOCK_HOP_FRAMES):
    chunk = X[:, :, start : start + BLOCK_FRAMES]
    doa.locate_sources(chunk, freq_range=FREQ_RANGE_HZ)
    block_power = np.asarray(doa.grid.values).real
    block_power = (block_power - block_power.min()) / (block_power.max() - block_power.min() + 1e-12)
    power += block_power
    num_blocks += 1

power /= max(num_blocks, 1)
peak_idx = select_distinct_peaks(
    power,
    AZIMUTHS_DEG,
    num_peaks=4,
    min_separation_deg=30.0,
    min_percentile=85.0,
)
detected_angles = AZIMUTHS_DEG[peak_idx]
detected_cardinals = [nearest_cardinal(angle) for angle in detected_angles]

print(f"  Aggregated {num_blocks} MUSIC blocks")
print(f"  Detected peaks: {[f'{angle:.1f}°' for angle in detected_angles]}")
print(f"  Nearest cardinals: {[f'{card}°' for card in detected_cardinals]}")


# ── Cartesian plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(AZIMUTHS_DEG, power, linewidth=1.0, color="darkorange")
for expected in [0, 90, 180, 270]:
    ax.axvline(expected, color="crimson", linestyle="--", alpha=0.6,
               label="expected" if expected == 0 else None)
for idx in peak_idx:
    ax.axvline(AZIMUTHS_DEG[idx], color="#0f766e", linestyle=":", alpha=0.9)
ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("MUSIC pseudo-spectrum")
ax.set_title(
    "Block-aggregated MUSIC spatial spectrum — example_mixture.wav"
    f"  ({BLOCK_FRAMES} STFT frames, num_src={ACTIVE_SOURCES_PER_BLOCK})"
)
ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax.legend()
plt.tight_layout()
cart_path = os.path.join(PLOT_DIR, "music_cartesian.png")
plt.savefig(cart_path, dpi=150)
plt.close()
print(f"Saved: {cart_path}")


# ── Polar plot ────────────────────────────────────────────────────────────────
power_norm = (power - power.min()) / (power.max() - power.min() + 1e-12)
az_rad = np.deg2rad(AZIMUTHS_DEG)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(1)
ax.plot(az_rad, power_norm, linewidth=1.2, color="darkorange")
ax.fill(az_rad, power_norm, alpha=0.25, color="darkorange")

for expected_az in [0, 90, 180, 270]:
    ax.axvline(np.deg2rad(expected_az), color="crimson", linestyle="--", linewidth=1, alpha=0.7)

for idx in peak_idx:
    pa = np.deg2rad(AZIMUTHS_DEG[idx])
    ax.plot([pa], [power_norm[idx]], marker="o", markersize=5, color="#0f766e")
    ax.text(pa, min(1.08, power_norm[idx] + 0.08), f"{AZIMUTHS_DEG[idx]:.1f}°",
            color="#0f766e", fontsize=9, ha="center", va="center")

ax.set_thetagrids([0, 90, 180, 270],
                  labels=["Front\n0°", "Left\n90°", "Back\n180°", "Right\n270°"],
                  fontsize=10, color="#c0392b")
ax.set_rticks([])
ax.set_title("Block-aggregated MUSIC spatial spectrum — example_mixture.wav", pad=24)
plt.tight_layout()
polar_path = os.path.join(PLOT_DIR, "music_polar.png")
plt.savefig(polar_path, dpi=150)
plt.close()
print(f"Saved: {polar_path}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n════════════════════════════════════════════════════════════════════════")
print("  Block-aggregated MUSIC — example_mixture.wav")
print(f"  nfft={NFFT}, hop={HOP}, block_frames={BLOCK_FRAMES}, active_sources={ACTIVE_SOURCES_PER_BLOCK}")
print(f"  Frequency range: {FREQ_RANGE_HZ[0]:.0f}–{FREQ_RANGE_HZ[1]:.0f} Hz")
print(f"  Detected peaks: {[f'{angle:.1f}°' for angle in detected_angles]}")
print(f"  Nearest cardinals: {[f'{card}°' for card in detected_cardinals]}")
print("  Expected: ['0.0°', '90.0°', '180.0°', '270.0°']")
print("════════════════════════════════════════════════════════════════════════")
