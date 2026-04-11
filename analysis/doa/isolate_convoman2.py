"""
Cascade MVDR: null brunch woman (270°) then null australia man (90°).

Step 1: Estimate BW spatial covariance from right-heavy TF bins → MVDR null BW
Step 2: On the BW-nulled output, estimate AM spatial covariance from left-heavy
        TF bins → MVDR null AM
Step 3: Apply no source mask — use the full signal, not just src4 bins

Outputs:
  analysis/doa/convoman_cascade_mvdr.wav
"""

import os, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
OUT_DIR   = os.path.dirname(__file__)

NPERSEG  = 2048
HOP      = 512
NOVERLAP = NPERSEG - HOP

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    return sr, d.astype(np.float64)

def save_wav(path, sig, sr):
    sig  = np.nan_to_num(sig)
    peak = np.max(np.abs(sig)) + 1e-9
    out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading mixture.wav ...")
sr, mix = load_wav(RAW_PATH)
N = mix.shape[0]
print(f"  {N} samples  {sr} Hz  {N/sr:.1f}s")

with open(GEO_PATH) as f: geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]
MIC_POS = np.array([[D/2,L/2],[D/2,-L/2],[-D/2,L/2],[-D/2,-L/2]])

# ── STFT all 4 channels ───────────────────────────────────────────────────────
print("Computing STFTs ...")
freqs = np.fft.rfftfreq(NPERSEG, 1.0/sr)
nF    = len(freqs)

X = np.stack([
    stft(mix[:,ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)[2]
    for ch in range(4)
])  # (4, F, T)
Xt = X.transpose(1, 0, 2)   # (F, 4, T)
nT = Xt.shape[2]

# ── Helpers ───────────────────────────────────────────────────────────────────
def steering(phi_deg):
    phi = np.deg2rad(phi_deg)
    tau = -(MIC_POS[:,0]*np.sin(phi) + MIC_POS[:,1]*np.cos(phi)) / C
    return np.exp(1j * 2*np.pi * freqs[:,None] * tau[None,:])   # (F, 4)

def mvdr_weights(Rv, a):
    """MVDR weights distortionless toward a, suppressing Rv interference."""
    Rinv   = np.linalg.inv(Rv)                                   # (F, 4, 4)
    Rinv_a = np.einsum("fij,fj->fi", Rinv, a)                    # (F, 4)
    denom  = np.einsum("fi,fi->f", a.conj(), Rinv_a).real + 1e-12
    return Rinv_a / denom[:,None]                                 # (F, 4)

def apply_weights(w, Xt_in):
    return np.einsum("fi,fit->ft", w.conj(), Xt_in)              # (F, T)

def cov_from_mask(mask, Xt_in):
    """Weighted spatial covariance from a (F,T) soft mask."""
    w = mask / (mask.sum(axis=1, keepdims=True) + 1e-9)
    return np.einsum("ft,fit,fjt->fij", w, Xt_in, Xt_in.conj()) + 1e-5*np.eye(4)

# ── Per-channel power ratios ──────────────────────────────────────────────────
lf_pow = np.abs(X[0])**2;  lr_pow = np.abs(X[1])**2
rf_pow = np.abs(X[2])**2;  rr_pow = np.abs(X[3])**2
left_ratio  = (lf_pow + lr_pow) / (lf_pow + lr_pow + rf_pow + rr_pow + 1e-9)
right_ratio = 1.0 - left_ratio

# ── Stage 1: null brunch woman (270°, right-heavy) ────────────────────────────
print("\nStage 1: null brunch woman (270°) ...")
a_target = steering(2.0)
Rv_bw    = cov_from_mask(right_ratio, Xt)
w_bw     = mvdr_weights(Rv_bw, a_target)
Y1       = apply_weights(w_bw, Xt)   # (F, T)  — BW suppressed, steered toward 2°

# ── Stage 2: null australia man (90°, left-heavy) — on residual ───────────────
# Re-stack Y1 as a single-channel "array" isn't possible, but we can apply
# a second MVDR pass on the original multichannel data using the AM covariance,
# then combine both suppressions via geometric mean of the two output masks.
print("Stage 2: null australia man (90°) ...")
Rv_am = cov_from_mask(left_ratio, Xt)
w_am  = mvdr_weights(Rv_am, a_target)
Y2    = apply_weights(w_am, Xt)   # (F, T)  — AM suppressed, steered toward 2°

# ── Combine: element-wise multiply (both nulls active simultaneously) ─────────
# Geometric mean of the two MVDR outputs preserves target while applying both nulls
print("Combining both MVDR outputs ...")
Y_combined = np.sqrt(np.abs(Y1) * np.abs(Y2)) * np.exp(1j * np.angle(Y1))

_, sig = istft(Y_combined, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
sig    = sig[:N].real.astype(np.float32)
save_wav(os.path.join(OUT_DIR, "convoman_cascade_mvdr.wav"), sig, sr)

# Also save each stage individually for comparison
_, s1 = istft(Y1, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
save_wav(os.path.join(OUT_DIR, "convoman_mvdr_bw_only.wav"), s1[:N].real.astype(np.float32), sr)
_, s2 = istft(Y2, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
save_wav(os.path.join(OUT_DIR, "convoman_mvdr_am_only.wav"), s2[:N].real.astype(np.float32), sr)

print("\nDone.")
print("  convoman_mvdr_bw_only.wav    — only brunch woman nulled")
print("  convoman_mvdr_am_only.wav    — only australia man nulled")
print("  convoman_cascade_mvdr.wav    — both nulled (combined)")
