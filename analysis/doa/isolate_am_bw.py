"""
Isolate Australia man and Brunch woman using inter-ear balance masks.

Australia man  → left-dominant  TF bins (left_ratio > 0.5 + THRESH)
Brunch woman   → right-dominant TF bins (left_ratio < 0.5 - THRESH)

Reconstruct mono by applying the mask to the dominant-ear channels
and averaging them.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")

NPERSEG  = 2048
HOP      = 512
NOVERLAP = NPERSEG - HOP
THRESH   = 0.15   # same as isolate_convoman4.py

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
    stft(mix[:, ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)[2]
    for ch in range(4)
])  # (4, F, T)

lf_pow = np.abs(X[0])**2 + np.abs(X[1])**2   # left ear  (LF+LR)
rf_pow = np.abs(X[2])**2 + np.abs(X[3])**2   # right ear (RF+RR)
left_ratio = lf_pow / (lf_pow + rf_pow + 1e-9)

am_mask  = (left_ratio > (0.5 + THRESH)).astype(np.float64)   # australia man
bw_mask  = (left_ratio < (0.5 - THRESH)).astype(np.float64)   # brunch woman

print(f"  Australia man  mask coverage: {am_mask.mean()*100:.1f}% of TF bins")
print(f"  Brunch woman   mask coverage: {bw_mask.mean()*100:.1f}% of TF bins")

def reconstruct_mono(mask, channels):
    """Average istft of masked bins across given channel indices."""
    sigs = []
    for ch in channels:
        _, sig = istft(X[ch] * mask, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
        sigs.append(sig[:N].real)
    return np.mean(sigs, axis=0)

# Australia man is left-dominant → use left channels (0=LF, 1=LR)
am_mono = reconstruct_mono(am_mask, [0, 1])
# Brunch woman is right-dominant → use right channels (2=RF, 3=RR)
bw_mono = reconstruct_mono(bw_mask, [2, 3])

print("\nSaving ...")
save_wav(os.path.join(SEP_DIR, "australia_man_interear.wav"),  am_mono, sr)
save_wav(os.path.join(SEP_DIR, "brunch_woman_interear.wav"),   bw_mono, sr)

print("\nDone.")
