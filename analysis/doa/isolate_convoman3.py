"""
Multichannel Wiener Filter (MWF) to suppress australia man + brunch woman,
followed by FastMNMF2 n=6 on the cleaned 4-channel output.

Uses fmnmf2-opt-n5 source masks to estimate per-source spatial covariances
directly from the raw multichannel data. Much cleaner than channel-ratio heuristics.

MWF per freq bin:
  Rn = covariance of interference (AM + BW)
  Rs = covariance of target (everything else)
  W  = Rs * inv(Rs + Rn)           (4×4 Wiener matrix)
  Y  = W @ X                       (4-channel cleaned output)

Outputs:
  analysis/ica/separated/mixture_mwf_cleaned.wav   (4-channel)
  analysis/ica/separated/mixture_mwf_fmnmf2_n6_source_{1-6}.wav
"""

import os, json
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import stft, istft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")

NPERSEG  = 2048
HOP      = 512
NOVERLAP = NPERSEG - HOP
N_SRC    = 6
N_ITER   = 150

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

print("Computing STFTs ...")
freqs = np.fft.rfftfreq(NPERSEG, 1.0/sr)

X = np.stack([
    stft(mix[:,ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)[2]
    for ch in range(4)
])  # (4, F, T)
Xt = X.transpose(1, 0, 2)   # (F, 4, T)
nF, nT = Xt.shape[0], Xt.shape[2]

print("Loading fmnmf2-opt-n5 source masks ...")
S_pow = {}
for n in range(1, 6):
    _, s = load_wav(os.path.join(SEP_DIR, f"mixture_fmnmf2_opt_n5_source_{n}.wav"))
    _, _, Z = stft(s[:N], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    S_pow[n] = np.abs(Z)**2   # (F, T)

total_pow = sum(S_pow[n] for n in range(1, 6)) + 1e-9

# Soft masks — raise to power 2 for harder separation
alpha = 2.0
masks = {n: (S_pow[n] / total_pow) ** alpha for n in range(1, 6)}
# Renormalise so masks sum to 1 per bin
mask_sum = sum(masks[n] for n in range(1, 6)) + 1e-9
masks = {n: masks[n] / mask_sum for n in range(1, 6)}

# ── Estimate per-source spatial covariance ────────────────────────────────────
print("Estimating spatial covariances ...")

def source_cov(mask):
    """Rk[f] = sum_t mask[f,t] * x[f,t] x[f,t]^H  (normalised)"""
    w = mask / (mask.sum(axis=1, keepdims=True) + 1e-9)   # (F, T)
    return np.einsum("ft,fit,fjt->fij", w, Xt, Xt.conj())  # (F, 4, 4)

R = {n: source_cov(masks[n]) for n in range(1, 6)}

# Interference = australia man (2) + brunch woman (4)
Rn = R[2] + R[4]
# Target = everything else
Rs = R[1] + R[3] + R[5]

# ── Multichannel Wiener Filter ────────────────────────────────────────────────
print("Computing MWF matrices ...")
Rtotal = Rs + Rn + 1e-5 * np.eye(4)[None,:,:]   # (F, 4, 4)
Rtotal_inv = np.linalg.inv(Rtotal)               # (F, 4, 4)
W = np.einsum("fij,fjk->fik", Rs, Rtotal_inv)    # W = Rs * (Rs+Rn)^{-1}

print("Applying MWF to multichannel data ...")
Y = np.einsum("fij,fjt->fit", W, Xt)             # (F, 4, T)

# Reconstruct 4-channel time-domain signal
cleaned = np.zeros((N, 4), dtype=np.float64)
for ch in range(4):
    _, sig_ch = istft(Y[:, ch, :], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    cleaned[:, ch] = sig_ch[:N].real

save_wav(os.path.join(SEP_DIR, "mixture_mwf_cleaned.wav"), cleaned, sr)

# ── FastMNMF2 n=6 on MWF-cleaned signal ──────────────────────────────────────
print(f"\nRunning FastMNMF2 n={N_SRC}, iter={N_ITER} on MWF-cleaned signal ...")
analysis_win  = pra.hann(NPERSEG)
synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, HOP)

X_bss = pra.transform.stft.analysis(cleaned, NPERSEG, HOP, win=analysis_win)
Y_all = pra.bss.fastmnmf2(X_bss, n_src=N_SRC, n_iter=N_ITER,
                           n_components=6, mic_index="all")

print("\nSaving sources ...")
for k in range(N_SRC):
    mono = pra.transform.stft.synthesis(
        Y_all[:,:,:,k].mean(axis=0), NPERSEG, HOP, win=synthesis_win
    )[:N].real
    rms = np.sqrt(np.mean(mono**2))
    print(f"  Source {k+1}: RMS={rms:.4f}")
    save_wav(os.path.join(SEP_DIR, f"mixture_mwf_fmnmf2_n6_source_{k+1}.wav"), mono, sr)

print("\nDone.")
