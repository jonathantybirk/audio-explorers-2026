"""
Use source 3 from lat-sup FastMNMF2 n=5 as a convo woman VAD.
When she's silent → convo man is talking.
Extract those segments from the lateral-suppressed mixture (hard-masked).

Also tries: harder binary suppression mask + FastMNMF2 n=2 on convo-man turns only.
"""

import os, json
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
OUT_DIR   = os.path.dirname(__file__)

NPERSEG  = 1024
HOP      = 256
NOVERLAP = NPERSEG - HOP

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    return sr, d.astype(np.float64)

def save_wav(path, sig, sr):
    sig = np.nan_to_num(sig)
    peak = np.max(np.abs(sig)) + 1e-9
    out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading ...")
sr, mix = load_wav(RAW_PATH)
N = mix.shape[0]

# fmnmf2-opt-n5 sources for suppression masks
S_pow = {}
for n in range(1, 6):
    _, s = load_wav(os.path.join(SEP_DIR, f"mixture_fmnmf2_opt_n5_source_{n}.wav"))
    _, _, Z = stft(s[:N], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    S_pow[n] = np.abs(Z)**2

total_pow = sum(S_pow[n] for n in range(1, 6)) + 1e-9

# lat-sup BSS source 3 = convo woman VAD reference
_, cw_src = load_wav(os.path.join(SEP_DIR, "mixture_latsup_fmnmf2_n5_source_3.wav"))

# ── Convo woman VAD from source 3 energy ─────────────────────────────────────
print("\nComputing convo woman VAD from lat-sup source 3 ...")
FRAME = int(0.04 * sr)
HOP_V = int(0.01 * sr)

n_frames = (N - FRAME) // HOP_V
cw_energy = np.array([
    np.mean(cw_src[i*HOP_V : i*HOP_V + FRAME]**2)
    for i in range(n_frames)
])

# Smooth and threshold
cw_smooth   = uniform_filter1d(cw_energy, size=15)
cw_max      = np.percentile(cw_smooth, 95)
cw_active   = cw_smooth > 0.15 * cw_max   # she's speaking
cm_active   = ~cw_active                   # she's silent → he might be speaking

# Build sample-level gate (soft edges)
gate = np.zeros(N, dtype=np.float64)
for i in range(n_frames):
    if cm_active[i]:
        start = i * HOP_V
        end   = min(start + FRAME, N)
        gate[start:end] = 1.0

from scipy.ndimage import gaussian_filter1d
gate = gaussian_filter1d(gate, sigma=HOP_V // 2)

cm_frac = cm_active.mean()
print(f"  Convo woman active: {cw_active.mean()*100:.0f}%  "
      f"Her-silent frames: {cm_frac*100:.0f}%")

# ── Hard suppression mask for lateral sources ─────────────────────────────────
lateral_ratio = (S_pow[2] + S_pow[4]) / total_pow
hard_mask = (lateral_ratio < 0.3).astype(np.float64)   # binary: keep bins where lateral < 30%
print(f"  Hard mask keeps {hard_mask.mean()*100:.0f}% of TF bins")

# ── Apply hard mask to all 4 channels + gate to CM turns ─────────────────────
print("\nApplying hard lateral mask + convo woman gate ...")
cleaned_gated = np.zeros_like(mix)
for ch in range(4):
    _, _, Z_ch = stft(mix[:, ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Z_masked   = Z_ch * hard_mask
    _, sig_ch  = istft(Z_masked, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    sig_ch     = sig_ch[:N].real * gate
    cleaned_gated[:, ch] = sig_ch

save_wav(os.path.join(OUT_DIR, "convoman_hardmask_gated.wav"),
         cleaned_gated[:, 0], sr)   # save LF channel as mono reference

# ── FastMNMF2 n=2 on the hard-masked gated 4-channel signal ──────────────────
print("\nRunning FastMNMF2 n=2 on hard-masked + gated signal ...")
analysis_win  = pra.hann(2048)
synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, 512)

X = pra.transform.stft.analysis(cleaned_gated, 2048, 512, win=analysis_win)

Y_all = pra.bss.fastmnmf2(X, n_src=2, n_iter=150, n_components=6, mic_index="all")

for k in range(2):
    Y_k  = Y_all[:, :, :, k].mean(axis=0)
    mono = pra.transform.stft.synthesis(Y_k, 2048, 512, win=synthesis_win)
    mono = mono[:N].real
    rms  = np.sqrt(np.mean(mono**2))
    print(f"  Source {k+1}: RMS={rms:.4f}")
    save_wav(os.path.join(OUT_DIR, f"convoman_n2_source_{k+1}.wav"), mono, sr)

print("\nDone.")
