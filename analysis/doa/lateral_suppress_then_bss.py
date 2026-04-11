"""
Step 1: Suppress australia man (src2, 90°) and brunch woman (src4, 270°)
        from the raw 4-channel mixture using Wiener TF masks.
        The same scalar mask is applied to all 4 channels so inter-channel
        spatial relationships are preserved.

Step 2: Run FastMNMF2 n=5 on the cleaned 4-channel signal.

Outputs saved to analysis/ica/separated/:
  mixture_lateral_suppressed.wav          (4-channel preprocessed)
  mixture_latsup_fmnmf2_n5_source_{1-5}.wav
"""

import os, json
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import stft, istft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")

N_SRC    = 5
N_ITER   = 100
STFT_SIZE = 2048
HOP_SIZE  = 512
N_COMP   = 6

MASK_NPERSEG = 1024
MASK_HOP     = 256
MASK_NOVERLAP = MASK_NPERSEG - MASK_HOP

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    return sr, d.astype(np.float64)

def save_wav(path, sig, sr):
    if sig.ndim == 1:
        peak = np.max(np.abs(sig)) + 1e-9
        out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
        wavfile.write(path, sr, (out * 32767).astype(np.int16))
    else:
        # multichannel: normalise globally
        peak = np.max(np.abs(sig)) + 1e-9
        out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
        wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")

# ── Load mixture ──────────────────────────────────────────────────────────────
print("Loading mixture.wav ...")
sr, mix = load_wav(RAW_PATH)
N = mix.shape[0]
print(f"  {N} samples  {sr} Hz  {N/sr:.1f}s")

# ── Load fmnmf2-opt-n5 sources for Wiener masks ───────────────────────────────
print("Loading fmnmf2-opt-n5 sources ...")
S_pow = {}
for n in range(1, 6):
    _, s = load_wav(os.path.join(SEP_DIR, f"mixture_fmnmf2_opt_n5_source_{n}.wav"))
    _, _, Z = stft(s[:N], fs=sr, nperseg=MASK_NPERSEG, noverlap=MASK_NOVERLAP)
    S_pow[n] = np.abs(Z)**2

total_pow    = sum(S_pow[n] for n in range(1, 6)) + 1e-9
suppress_pow = S_pow[2] + S_pow[4]           # australia man + brunch woman
keep_mask    = np.clip(1.0 - suppress_pow / total_pow, 0.0, 1.0)  # (F, T)
print(f"  Suppression mask: mean={keep_mask.mean():.3f}  "
      f"fraction of bins kept fully: {(keep_mask > 0.9).mean():.2f}")

# ── Apply mask to all 4 channels ──────────────────────────────────────────────
print("Applying suppression mask to raw channels ...")
cleaned = np.zeros_like(mix)
for ch in range(4):
    _, _, Z_ch = stft(mix[:, ch], fs=sr, nperseg=MASK_NPERSEG, noverlap=MASK_NOVERLAP)
    Z_masked   = Z_ch * keep_mask
    _, sig_ch  = istft(Z_masked, fs=sr, nperseg=MASK_NPERSEG, noverlap=MASK_NOVERLAP)
    sig_ch = sig_ch[:N].real
    cleaned[:, ch] = sig_ch

out_4ch = os.path.join(SEP_DIR, "mixture_lateral_suppressed.wav")
save_wav(out_4ch, cleaned, sr)

# ── FastMNMF2 separation on cleaned signal ────────────────────────────────────
print(f"\nRunning FastMNMF2 n={N_SRC} on lateral-suppressed mixture ...")
print(f"  STFT={STFT_SIZE}  hop={HOP_SIZE}  iter={N_ITER}  n_comp={N_COMP}")

analysis_win  = pra.hann(STFT_SIZE)
synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, HOP_SIZE)

X = pra.transform.stft.analysis(cleaned, STFT_SIZE, HOP_SIZE, win=analysis_win)

Y_all = pra.bss.fastmnmf2(
    X,
    n_src=N_SRC,
    n_iter=N_ITER,
    n_components=N_COMP,
    mic_index="all",
)
# Y_all: (nchannels, nframes, nfrequencies, nsources)

print("\nSaving sources ...")
for k in range(N_SRC):
    Y_k   = Y_all[:, :, :, k].mean(axis=0)
    mono  = pra.transform.stft.synthesis(Y_k, STFT_SIZE, HOP_SIZE, win=synthesis_win)
    mono  = mono[:N].real

    rms = np.sqrt(np.mean(mono**2))
    print(f"  Source {k+1}: RMS={rms:.4f}")

    out_path = os.path.join(SEP_DIR, f"mixture_latsup_fmnmf2_n5_source_{k+1}.wav")
    save_wav(out_path, mono, sr)

print("\nDone.")
