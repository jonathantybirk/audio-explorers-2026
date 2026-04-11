"""
Convo man isolation pipeline.

Strategy: fmnmf2-opt-n5 source 4 co-isolates convo man + brunch woman.
Brunch woman is confirmed at 270°. Apply a spatial null toward 270° on the
raw 4-channel mixture, weighted by source 4's TF mask, so we focus only on
the bins where convo man / brunch woman are active — then kill brunch woman
spatially within those bins.

Outputs:
  analysis/doa/convoman_wiener.wav          — src4 mask + spatial null 270°
  analysis/doa/convoman_wiener_vad.wav      — + pitch VAD on top
"""

import json, os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
OUT_DIR   = os.path.dirname(__file__)

# ── I/O ───────────────────────────────────────────────────────────────────────
def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float32) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float32) / 2**31
    return sr, d.astype(np.float32)

def save_wav(path, sig, sr):
    peak = np.max(np.abs(sig)) + 1e-9
    out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")

# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f: geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
MIC_POS = np.array([[D/2, L/2],[D/2,-L/2],[-D/2, L/2],[-D/2,-L/2]], dtype=np.float32)

# ── Load raw mixture ──────────────────────────────────────────────────────────
print("Loading mixture.wav ...")
sr, mix = load_wav(RAW_PATH)
N = mix.shape[0]
channels = [mix[:, c] for c in range(4)]
print(f"  {N} samples  {sr} Hz  {N/sr:.1f}s")

# ── STFT params ───────────────────────────────────────────────────────────────
NPERSEG = 1024
HOP     = 256
NOVERLAP = NPERSEG - HOP

def ch_stft(x):
    _, _, Z = stft(x, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return Z  # (nFreqs, nFrames)

print("Computing STFTs of raw channels ...")
X = np.stack([ch_stft(c) for c in channels])   # (4, F, T)
nFreqs, nFrames = X.shape[1], X.shape[2]
freqs = np.fft.rfftfreq(NPERSEG, 1.0/sr)

# ── Load fmnmf2-opt-n5 sources ────────────────────────────────────────────────
print("Loading fmnmf2-opt-n5 sources ...")
srcs = {}
for n in range(1, 6):
    _, s = load_wav(os.path.join(SEP_DIR, f"mixture_fmnmf2_opt_n5_source_{n}.wav"))
    srcs[n] = s

def src_stft_pow(x):
    _, _, Z = stft(x[:N], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return np.abs(Z)**2

print("Computing source power spectrograms ...")
S_pow     = {n: src_stft_pow(srcs[n]) for n in range(1, 6)}
total_pow = sum(S_pow[n] for n in range(1, 6)) + 1e-9

# TF mask: bins where source 4 (convo man + brunch woman) dominates
src4_mask = S_pow[4] / total_pow          # (F, T)  high where convo man/BW active

# ── Mask-MVDR: estimate brunch woman's spatial covariance, suppress her ────────
print("Mask-MVDR: estimating brunch woman spatial covariance ...")
Xt = X.transpose(1, 0, 2)                              # (F, 4, T)

# Right-channel dominance mask: brunch woman is at 270° so RF/RR >> LF/LR
rf_pow = np.abs(X[2])**2   # RF
rr_pow = np.abs(X[3])**2   # RR
lf_pow = np.abs(X[0])**2   # LF
lr_pow = np.abs(X[1])**2   # LR
right_ratio = (rf_pow + rr_pow) / (lf_pow + lr_pow + rf_pow + rr_pow + 1e-9)

# Brunch woman bins: src4 active AND right-heavy
bw_mask = src4_mask * right_ratio          # (F, T) — high where BW dominates

# Convo man bins: src4 active AND NOT right-heavy (or less right-heavy)
cm_mask = src4_mask * (1.0 - right_ratio)  # (F, T) — high where CM more likely

print(f"  BW mask mean={bw_mask.mean():.4f}  CM mask mean={cm_mask.mean():.4f}")

# Estimate brunch woman interference covariance Rv from BW-heavy bins
# Rv[f] = sum_t bw_mask[f,t] * x[f,t] * x[f,t]^H  (weighted outer product)
print("  Computing interference covariance ...")
bw_weights = bw_mask / (bw_mask.sum(axis=1, keepdims=True) + 1e-9)  # normalise per freq
Rv = np.einsum("ft,fit,fjt->fij", bw_weights, Xt, Xt.conj())        # (F, 4, 4)
Rv += 1e-4 * np.eye(4)[None,:,:]                                      # diagonal loading

# Steering vector toward ~2° (convo man / front)
phi_cm = np.deg2rad(2.0)
tau_cm = -(MIC_POS[:,0]*np.sin(phi_cm) + MIC_POS[:,1]*np.cos(phi_cm)) / C
a_cm   = np.exp(1j * 2*np.pi * freqs[:,None] * tau_cm[None,:])       # (F, 4)

# MVDR weights: w = Rv^{-1} a / (a^H Rv^{-1} a)
print("  Inverting covariance matrices ...")
Rinv   = np.linalg.inv(Rv)                                            # (F, 4, 4)
Rinv_a = np.einsum("fij,fj->fi", Rinv, a_cm)                         # (F, 4)
denom  = np.einsum("fi,fi->f", a_cm.conj(), Rinv_a).real + 1e-12     # (F,)
w      = Rinv_a / denom[:,None]                                       # (F, 4)

# Apply MVDR weights, then weight output by CM mask
Y = np.einsum("fi,fit->ft", w.conj(), Xt)    # (F, T)
Y_masked = Y * cm_mask

_, sig_wiener = istft(Y_masked, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
sig_wiener = sig_wiener[:N].real.astype(np.float32)
save_wav(os.path.join(OUT_DIR, "convoman_wiener.wav"), sig_wiener, sr)

# ── Pitch-based VAD ───────────────────────────────────────────────────────────
print("\nPitch-based VAD ...")

# Use autocorrelation on short frames to estimate F0
FRAME_LEN  = int(0.04 * sr)    # 40ms
FRAME_HOP  = int(0.01 * sr)    # 10ms hop
F0_MIN = 80    # Hz — lowest male pitch
F0_MAX = 320   # Hz — highest female pitch
FEMALE_THRESH = 170  # Hz — above this → likely female

def estimate_f0(frame):
    """Autocorrelation-based F0 estimate. Returns Hz or 0 if unvoiced."""
    frame = frame * np.hanning(len(frame))
    corr  = np.correlate(frame, frame, mode="full")
    corr  = corr[len(corr)//2:]           # keep positive lags
    lag_min = int(sr / F0_MAX)
    lag_max = int(sr / F0_MIN)
    if lag_max >= len(corr): return 0.0
    peak_lag = np.argmax(corr[lag_min:lag_max]) + lag_min
    # voiced check: peak must be reasonably strong
    if corr[peak_lag] < 0.15 * corr[0] + 1e-9:
        return 0.0
    return sr / peak_lag

# Estimate F0 per frame on the Wiener-suppressed output
n_frames_vad = (N - FRAME_LEN) // FRAME_HOP + 1
f0s    = np.zeros(n_frames_vad)
voiced = np.zeros(n_frames_vad, dtype=bool)
female = np.zeros(n_frames_vad, dtype=bool)

for i in range(n_frames_vad):
    start = i * FRAME_HOP
    frame = sig_wiener[start:start + FRAME_LEN]
    if len(frame) < FRAME_LEN:
        break
    f0 = estimate_f0(frame)
    f0s[i]    = f0
    voiced[i] = f0 > 0
    female[i] = f0 > FEMALE_THRESH

# Smooth female mask over ±3 frames
from scipy.ndimage import uniform_filter1d
female_smooth = uniform_filter1d(female.astype(float), size=7)
female_gate   = female_smooth > 0.4   # suppress if majority female in neighbourhood

print(f"  Voiced frames:  {voiced.sum()} / {n_frames_vad}  ({100*voiced.mean():.0f}%)")
print(f"  Female frames:  {female.sum()} / {n_frames_vad}  ({100*female.mean():.0f}%)")
print(f"  Male-ish frames kept: {(~female_gate & voiced).sum()}")

# Build sample-level gate
gate = np.ones(N, dtype=np.float32)
for i in range(n_frames_vad):
    start = i * FRAME_HOP
    end   = min(start + FRAME_LEN, N)
    if female_gate[i]:
        # Soft suppress female-dominated frames (keep 10% to avoid hard clicks)
        gate[start:end] = np.minimum(gate[start:end], 0.1)

# Smooth gate edges with a short hanning blend
from scipy.ndimage import gaussian_filter1d
gate = gaussian_filter1d(gate, sigma=FRAME_HOP//2)

sig_vad = sig_wiener * gate
save_wav(os.path.join(OUT_DIR, "convoman_wiener_vad.wav"), sig_vad, sr)

print("\nDone.")
print("  convoman_wiener.wav     — lateral suppression only")
print("  convoman_wiener_vad.wav — + pitch VAD (female frames suppressed)")
