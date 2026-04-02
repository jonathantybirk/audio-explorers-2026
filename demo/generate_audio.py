"""
Generate beamformed audio outputs for the demo player.

Run from repo root:
    python demo/generate_audio.py

Outputs saved to demo/audio/:

  By direction (D&S and MVDR steered toward each known talker position):
    das_{direction}.wav
    mvdr_{direction}.wav

  By microphone (raw single-channel extracts):
    mic_LF.wav   mic_LR.wav   mic_RF.wav   mic_RR.wav

Geometry loaded from data/mic_geometry.json (produced by
analysis/microphone_geometry/joint_geometry_fit.py).
Talker positions (0°/90°/180°/270°) are ground truth for example_mixture.wav
only — see setup.md.
"""

import os
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

# Channel order per case spec: LF=0, LR=1, RF=2, RR=3
MIC_POS = np.array([
    [ D/2,  L/2],   # LF
    [ D/2, -L/2],   # LR
    [-D/2,  L/2],   # RF
    [-D/2, -L/2],   # RR
])
MIC_LABELS = ["LF", "LR", "RF", "RR"]

# Known talker positions in example_mixture (ground truth from case PDF)
DIRECTIONS = {
    "0deg_front":   0,
    "90deg_left":   90,
    "180deg_back":  180,
    "270deg_right": 270,
}

NPERSEG = 512
HOP     = 256


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    return sr, data.astype(np.float32)


def save_wav(path, signal, sr):
    peak = np.max(np.abs(signal)) + 1e-9
    out  = np.clip(signal / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"    saved  {os.path.relpath(path)}")


# ── Beamformers ───────────────────────────────────────────────────────────────
def delay_and_sum(channels, phi, sr):
    """
    Time-domain Delay-and-Sum with fractional-sample linear interpolation.
    Works for any azimuth phi (radians). Not limited to cardinal directions.
    """
    N   = len(channels[0])
    out = np.zeros(N, dtype=np.float32)
    for m in range(4):
        tau  = -(MIC_POS[m, 0]*np.sin(phi) + MIC_POS[m, 1]*np.cos(phi)) / C
        d    = tau * sr
        lo   = int(np.floor(d))
        frac = d - lo
        idx  = np.arange(N) + lo
        v    = (idx >= 0) & (idx < N - 1)
        s    = np.zeros(N, dtype=np.float32)
        s[v] = (1-frac)*channels[m][idx[v]] + frac*channels[m][idx[v]+1]
        out += s
    return out / 4.0


def mvdr(channels, phi, sr):
    """
    MVDR beamformer with blocking matrix (interference-only covariance).

    The standard MVDR pitfall: estimating covariance from the full mixture
    includes the target signal, biasing the weights and partially cancelling
    what you want to keep.  Fix: apply a blocking matrix B = I - aa^H/(a^Ha)
    that projects out the target direction from the data, then estimate the
    interference covariance from the residual.  Weights are still constrained
    to be distortionless toward the target (MVDR constraint).

    Works for any azimuth phi (radians).
    """
    freqs = np.fft.rfftfreq(NPERSEG, 1.0 / sr)

    X  = np.stack([
        stft(ch, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)[2]
        for ch in channels
    ])                                  # (4, nFreqs, nFrames)
    Xt = X.transpose(1, 0, 2)          # (nFreqs, 4, nFrames)
    n_freqs, n_mics, n_frames = Xt.shape

    # Steering vector a[f, m] = exp(j 2π f τ_m)
    tau = -(MIC_POS[:, 0]*np.sin(phi) + MIC_POS[:, 1]*np.cos(phi)) / C
    a   = np.exp(1j * 2*np.pi * freqs[:, None] * tau[None, :])  # (nFreqs, 4)

    # Blocking matrix B[f] = I - a a^H / (a^H a)
    # Projects OUT the target direction; blocked data contains interference only
    a_energy = np.einsum("fi,fi->f", a.conj(), a).real             # (nFreqs,)
    B = (np.eye(n_mics)[None, :, :]
         - np.einsum("fi,fj->fij", a, a.conj()) / a_energy[:, None, None])

    # Interference-only data and covariance
    Zt = np.einsum("fij,fjt->fit", B, Xt)                          # (nFreqs, 4, nFrames)
    Rv = np.einsum("fit,fjt->fij", Zt, Zt.conj()) / n_frames
    Rv += 1e-4 * np.eye(n_mics)[None, :, :]

    # MVDR weights: w = Rv^{-1} a / (a^H Rv^{-1} a)
    Rinv   = np.linalg.inv(Rv)
    Rinv_a = np.einsum("fij,fj->fi", Rinv, a)
    denom  = np.einsum("fi,fi->f", a.conj(), Rinv_a)
    w      = Rinv_a / (denom[:, None] + 1e-12)

    Y = np.einsum("fi,fit->ft", w.conj(), Xt)

    _, out = istft(Y, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    N      = len(channels[0])
    out    = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
    return out.real.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} …")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
      f"|  {sr} Hz  |  {data.shape[0]/sr:.1f} s\n")

channels = [data[:, ch] for ch in range(4)]

# Unprocessed reference — simple average of all 4 channels (omnidirectional, no steering)
print("  reference mix (all 4 channels averaged)")
save_wav(os.path.join(OUT_DIR, "reference_mix.wav"),
         np.mean(np.stack(channels), axis=0), sr)

# Per-direction beamformed outputs
for dir_name, az_deg in DIRECTIONS.items():
    phi = np.deg2rad(az_deg)
    print(f"  {dir_name}  ({az_deg}°)")
    save_wav(os.path.join(OUT_DIR, f"das_{dir_name}.wav"),
             delay_and_sum(channels, phi, sr), sr)
    save_wav(os.path.join(OUT_DIR, f"mvdr_{dir_name}.wav"),
             mvdr(channels, phi, sr), sr)

# Raw single-channel extracts (one per microphone)
print("\n  Raw microphone channels")
for ch, label in enumerate(MIC_LABELS):
    save_wav(os.path.join(OUT_DIR, f"mic_{label}.wav"), channels[ch], sr)

print(f"\nDone — {4*2 + 4} files written to {os.path.relpath(OUT_DIR)}/")
