"""
Apply Australia man inter-ear mask to all 4 channels, then run FastMNMF2 n=2.
Australia man is left-dominant (left_ratio > 0.5 + THRESH).
Preserving 4-channel spatial info before BSS gives much cleaner separation.
"""
import os
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
THRESH   = 0.15
N_SRC    = 2
N_ITER   = 200

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

lf_pow     = np.abs(X[0])**2 + np.abs(X[1])**2
rf_pow     = np.abs(X[2])**2 + np.abs(X[3])**2
left_ratio = lf_pow / (lf_pow + rf_pow + 1e-9)
am_mask    = (left_ratio > (0.5 + THRESH)).astype(np.float64)

print(f"  Australia man mask: {am_mask.mean()*100:.1f}% of TF bins kept")

print("Applying mask to all 4 channels ...")
cleaned = np.zeros((N, 4), dtype=np.float64)
for ch in range(4):
    _, sig = istft(X[ch] * am_mask, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    cleaned[:, ch] = sig[:N].real

print(f"Running FastMNMF2 n={N_SRC}, iter={N_ITER} ...")
aw = pra.hann(NPERSEG)
sw = pra.transform.stft.compute_synthesis_window(aw, HOP)
X_bss = pra.transform.stft.analysis(cleaned, NPERSEG, HOP, win=aw)
Y_all = pra.bss.fastmnmf2(X_bss, n_src=N_SRC, n_iter=N_ITER,
                           n_components=6, mic_index="all")

print("Saving sources ...")
for k in range(N_SRC):
    mono = pra.transform.stft.synthesis(
        Y_all[:,:,:,k].mean(axis=0), NPERSEG, HOP, win=sw
    )[:N].real
    rms = np.sqrt(np.mean(mono**2))
    print(f"  Source {k+1}: RMS={rms:.4f}")
    save_wav(os.path.join(SEP_DIR, f"australia_man_bss_n2_source_{k+1}.wav"), mono, sr)

print("\nDone.")
