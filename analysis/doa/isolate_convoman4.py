"""
Simple inter-ear balance kill:
  - TF bin heard mostly in RIGHT channels (RF+RR >> LF+LR) → brunch woman → kill
  - TF bin heard mostly in LEFT channels  (LF+LR >> RF+RR) → australia man → kill
  - TF bin heard in BOTH → keep

Apply this binary mask to all 4 channels (preserving spatial info), run FastMNMF2 n=6.

Threshold: kill if |left_ratio - 0.5| > THRESH  (i.e. >65% one side)
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
THRESH   = 0.15   # kill if left_ratio < 0.35 or > 0.65
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

print("Loading mixture.wav ...")
sr, mix = load_wav(RAW_PATH)
N = mix.shape[0]

print("Computing STFTs ...")
X = np.stack([
    stft(mix[:,ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)[2]
    for ch in range(4)
])  # (4, F, T)

# ── Inter-ear balance mask ────────────────────────────────────────────────────
lf_pow = np.abs(X[0])**2 + np.abs(X[1])**2   # left ear  (LF+LR)
rf_pow = np.abs(X[2])**2 + np.abs(X[3])**2   # right ear (RF+RR)
left_ratio = lf_pow / (lf_pow + rf_pow + 1e-9)   # 0=all right, 1=all left

# Keep bins heard in both ears; kill laterally dominated bins
keep = (left_ratio >= (0.5 - THRESH)) & (left_ratio <= (0.5 + THRESH))
mask = keep.astype(np.float64)   # (F, T)

killed_left  = (left_ratio > (0.5 + THRESH)).mean()
killed_right = (left_ratio < (0.5 - THRESH)).mean()
kept         = mask.mean()
print(f"  Killed left-dominant (australia man):  {killed_left*100:.1f}% of TF bins")
print(f"  Killed right-dominant (brunch woman):  {killed_right*100:.1f}% of TF bins")
print(f"  Kept (heard in both):                  {kept*100:.1f}% of TF bins")

# ── Apply mask to all 4 channels ─────────────────────────────────────────────
print("Applying inter-ear balance mask to all 4 channels ...")
cleaned = np.zeros((N, 4), dtype=np.float64)
for ch in range(4):
    Z_masked = X[ch] * mask
    _, sig   = istft(Z_masked, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    cleaned[:, ch] = sig[:N].real

save_wav(os.path.join(SEP_DIR, "mixture_interear_masked.wav"), cleaned, sr)

# ── FastMNMF2 n=6 ─────────────────────────────────────────────────────────────
print(f"\nRunning FastMNMF2 n={N_SRC}, iter={N_ITER} ...")
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
    save_wav(os.path.join(SEP_DIR, f"mixture_interear_fmnmf2_n6_source_{k+1}.wav"), mono, sr)

print("\nDone.")
