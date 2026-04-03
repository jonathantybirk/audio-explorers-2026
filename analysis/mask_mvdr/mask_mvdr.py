"""
Mask-based MVDR beamforming on a 4-channel mixture.

Classical MVDR (as in demo/generate_audio.py) uses a blocking matrix to
estimate the interference covariance, but still needs oracle steering angles
and the small front-back TDoA limits steering accuracy.

This script instead uses ILRMA to compute per-source time-frequency masks and
uses those masks to build accurate spatial covariance matrices per source.
The MVDR beamformer then suppresses interference while keeping the target
signal distortionless.

This is the classical state-of-the-art pipeline and what commercial hearing
aids do in simplified form.  No geometry or steering angles are needed for the
covariance estimation step — only for the final MVDR steering constraint.

Pipeline:
  1. ILRMA separation → per-source TF masks.
  2. For each target source k:
       Rk  = (mask_k weighted) spatial covariance of the mixture (target + interference)
       Rvk = (1 - mask_k weighted) covariance (interference only)
  3. MVDR weights: w_k = Rvk^{-1} * a_k / (a_k^H * Rvk^{-1} * a_k)
     where a_k is the steering vector toward source k's estimated DoA.
  4. Apply w_k to the mixture STFT to get the beamformed signal.

Outputs saved to analysis/mask_mvdr/beamformed/

── Hyperparameters ──────────────────────────────────────────────────────────
All hyperparameters are at the top of this file.
"""

import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import istft, stft

# ── Recording to process ──────────────────────────────────────────────────────
WAV_KEY = "example"   # "example" or "mixture"

# ── ILRMA hyperparameters (used for mask estimation) ─────────────────────────
STFT_SIZE      = 2048
HOP_SIZE       = 1024
ILRMA_ITERS    = 100
NMF_COMPONENTS = 4

# ── MVDR hyperparameters ──────────────────────────────────────────────────────
DIAG_LOAD = 1e-4   # diagonal loading to regularise covariance inversion

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav"),
}
WAV_PATH = _WAV_PATHS[WAV_KEY]
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "beamformed")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)

D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [ D/2,  L/2],
    [ D/2, -L/2],
    [-D/2,  L/2],
    [-D/2, -L/2],
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    return sr, data.astype(np.float64)


def save_wav(path, signal, sr):
    peak = np.max(np.abs(signal)) + 1e-12
    out = np.clip(signal / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")


# ── Spatial helpers ───────────────────────────────────────────────────────────
def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa(ch_a, ch_b, phi_rad):
    return (
        (MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]) * np.sin(phi_rad)
        + (MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]) * np.cos(phi_rad)
    ) / C


def srp_phat_spectrum(channels, sr, azimuths):
    n     = min(len(ch) for ch in channels)
    n_fft = 1 << (n - 1).bit_length()
    channels = [ch[:n] for ch in channels]
    gcc_store = {(a, b): gcc_phat(channels[a], channels[b], n_fft)
                 for a, b in ALL_PAIRS}
    power = np.zeros(len(azimuths), dtype=np.float64)
    for idx, az in enumerate(azimuths):
        phi = np.deg2rad(az)
        s = 0.0
        for ch_a, ch_b in ALL_PAIRS:
            tau = predicted_tdoa(ch_a, ch_b, phi)
            lag_idx = tau * sr + n_fft // 2
            lo = int(np.floor(lag_idx)); hi = lo + 1
            frac = lag_idx - lo
            lo = np.clip(lo, 0, n_fft - 1); hi = np.clip(hi, 0, n_fft - 1)
            s += (1 - frac) * gcc_store[(ch_a, ch_b)][lo] + frac * gcc_store[(ch_a, ch_b)][hi]
        power[idx] = s
    return power


def nearest_cardinal(angle_deg):
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda ref: abs(((angle_deg - ref + 180) % 360) - 180))


def cardinal_key(angle_deg):
    return {0: "0deg_front", 90: "90deg_left",
            180: "180deg_back", 270: "270deg_right"}[angle_deg]


def steering_vector(freqs_hz, doa_deg):
    """Far-field steering vector a[f, m] = exp(j 2π f τ_m) for azimuth doa_deg."""
    phi = np.deg2rad(doa_deg)
    tau = -(MIC_POS[:, 0] * np.sin(phi) + MIC_POS[:, 1] * np.cos(phi)) / C
    return np.exp(1j * 2 * np.pi * freqs_hz[:, None] * tau[None, :])  # (F, M)


# ── ILRMA step ────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
      f"|  {sr} Hz  |  {data.shape[0]/sr:.1f} s\n")

analysis_win  = pra.hann(STFT_SIZE)
synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, HOP_SIZE)

print(f"Running ILRMA (iter={ILRMA_ITERS}, NMF_k={NMF_COMPONENTS}) for mask estimation ...")
X_pra = pra.transform.stft.analysis(data, STFT_SIZE, HOP_SIZE, win=analysis_win)
# X_pra: (nframes, nfreqs, nchannels)

Y_pra, W_pra = pra.bss.ilrma(
    X_pra, n_src=data.shape[1], n_iter=ILRMA_ITERS,
    proj_back=False, n_components=NMF_COMPONENTS, return_filters=True,
)
# Y_pra: (nframes, nfreqs, nsrc)

# Reconstruct mixing matrix A = inv(W)
A_pra = np.linalg.inv(W_pra.transpose(0, 2, 1)).transpose(0, 2, 1)
# A_pra: (nfreqs, nchannels, nsrc)

print(f"  STFT shape: {X_pra.shape}")

# ── Estimate DoA of each ILRMA source via SRP-PHAT ────────────────────────────
print("\nEstimating DoA of each ILRMA source ...")
azimuths = np.linspace(0, 360, 720, endpoint=False)
source_doas = []
for k in range(Y_pra.shape[2]):
    # Reconstruct source image on all mics
    img_stft = Y_pra[:, :, k][:, :, None] * A_pra[:, :, k][None, :, :]
    channels_k = []
    for m in range(img_stft.shape[2]):
        sig = pra.transform.stft.synthesis(img_stft[:, :, m], STFT_SIZE, HOP_SIZE, win=synthesis_win)
        channels_k.append(sig[:data.shape[0]].real.astype(np.float64))
    power = srp_phat_spectrum(channels_k, sr, azimuths)
    doa   = float(azimuths[np.argmax(power)])
    source_doas.append(doa)
    print(f"  Source {k}: DoA = {doa:.1f}°  →  nearest cardinal {nearest_cardinal(doa)}°")

# ── Build TF masks from ILRMA source energies ─────────────────────────────────
# Soft mask: mask_k[t, f] = |Y_k[t,f]|^2 / sum_j |Y_j[t,f]|^2
Y_power    = np.abs(Y_pra) ** 2                       # (T, F, nsrc)
total_power = Y_power.sum(axis=2, keepdims=True) + 1e-12
masks       = Y_power / total_power                   # (T, F, nsrc)  soft Wiener-like masks

# ── MVDR beamforming using mask-estimated covariances ─────────────────────────
# Recompute STFT in (F, M, T) layout for MVDR math
freqs_hz = np.fft.rfftfreq(STFT_SIZE, 1.0 / sr)      # (F,)
M = data.shape[1]
F = len(freqs_hz)

# Stack raw mixture STFT in (F, M, T) for covariance math
X_fmt = X_pra.transpose(1, 2, 0)    # (F, M, T)

print("\nApplying mask-MVDR beamforming ...")
beamformed = {}
for k in range(Y_pra.shape[2]):
    doa   = source_doas[k]
    card  = nearest_cardinal(doa)
    a     = steering_vector(freqs_hz, doa)        # (F, M)
    mk    = masks[:, :, k].T                      # (F, T) — target mask for source k
    mi    = 1.0 - mk                              # (F, T) — interference mask

    # Target and interference covariances estimated via weighted outer products
    Rk  = np.einsum("ft,fmi,fni->fmn", mk, X_fmt, X_fmt.conj()) / X_fmt.shape[2]
    Rv  = np.einsum("ft,fmi,fni->fmn", mi, X_fmt, X_fmt.conj()) / X_fmt.shape[2]
    Rv += DIAG_LOAD * np.eye(M)[None, :, :]       # diagonal loading

    # MVDR weights: w = Rv^{-1} a / (a^H Rv^{-1} a)
    Rinv   = np.linalg.inv(Rv)                    # (F, M, M)
    Rinv_a = np.einsum("fmn,fn->fm", Rinv, a)    # (F, M)
    denom  = np.einsum("fm,fm->f", a.conj(), Rinv_a).real + 1e-12
    w      = Rinv_a / denom[:, None]              # (F, M)

    # Apply beamformer to mixture STFT
    Y_beam = np.einsum("fm,fmt->ft", w.conj(), X_fmt)   # (F, T)

    # ISTFT
    _, out = istft(Y_beam, fs=sr, nperseg=STFT_SIZE, noverlap=STFT_SIZE - HOP_SIZE)
    N = data.shape[0]
    out = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
    beamformed[(k, card, doa)] = out.real.astype(np.float64)
    print(f"  Source {k} ({doa:.1f}°, nearest {card}°) — beamformed")

print("\nSaving beamformed audio ...")
for (k, card, doa), signal in beamformed.items():
    save_wav(os.path.join(OUT_DIR, f"mmvdr_source_{k+1}_{doa:.0f}deg.wav"), signal, sr)
    save_wav(os.path.join(OUT_DIR, f"mmvdr_{cardinal_key(card)}.wav"), signal, sr)

# ── Spectrograms ──────────────────────────────────────────────────────────────
print("\nPlotting spectrograms ...")
order = list(np.argsort(source_doas))
entries = list(beamformed.items())  # same order as source_doas
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for rank, k in enumerate(order):
    ax    = axes[rank // 2][rank % 2]
    _, card, doa = list(beamformed.keys())[k]
    sig   = list(beamformed.values())[k]
    ax.specgram(sig, Fs=sr, NFFT=512, noverlap=256, cmap="cividis")
    ax.set_title(f"Mask-MVDR source {rank+1}  —  {doa:.1f}°  (nearest {card}°)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
plt.suptitle("Mask-based MVDR beamformed sources — " + os.path.basename(WAV_PATH), fontsize=13)
plt.tight_layout()
spec_path = os.path.join(OUT_DIR, "mmvdr_spectrograms.png")
plt.savefig(spec_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(spec_path)}")

print("\n════════════════════════════════════════════════════════════════════════")
print("  Mask-based MVDR — " + os.path.basename(WAV_PATH))
print(f"  ILRMA: STFT={STFT_SIZE}, hop={HOP_SIZE}, iter={ILRMA_ITERS}, NMF_k={NMF_COMPONENTS}")
print(f"  MVDR: diagonal loading={DIAG_LOAD}")
print("  Estimated source directions:")
for rank, k in enumerate(order):
    print(f"    Source {rank+1}: {source_doas[k]:.1f}°  (nearest {nearest_cardinal(source_doas[k])}°)")
print("════════════════════════════════════════════════════════════════════════")
