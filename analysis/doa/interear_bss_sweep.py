"""
Run FastMNMF2 with n=3,4,5,6 on the inter-ear balance masked mixture.
"""
import os
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
IN_PATH   = os.path.join(SEP_DIR, "mixture_interear_masked.wav")

NPERSEG = 2048
HOP     = 512
N_ITER  = 150
N_COMP  = 6

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
    print(f"    saved  {os.path.relpath(path)}")

print("Loading inter-ear masked mixture ...")
sr, data = load_wav(IN_PATH)
N = data.shape[0]
print(f"  {N} samples  {sr} Hz  {N/sr:.1f}s\n")

analysis_win  = pra.hann(NPERSEG)
synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, HOP)
X = pra.transform.stft.analysis(data, NPERSEG, HOP, win=analysis_win)

for n_src in [3, 4, 5, 6]:
    print(f"{'='*60}")
    print(f"FastMNMF2 n={n_src}  iter={N_ITER}  n_comp={N_COMP}")
    print(f"{'='*60}")

    Y_all = pra.bss.fastmnmf2(X.copy(), n_src=n_src, n_iter=N_ITER,
                               n_components=N_COMP, mic_index="all")

    for k in range(n_src):
        mono = pra.transform.stft.synthesis(
            Y_all[:,:,:,k].mean(axis=0), NPERSEG, HOP, win=synthesis_win
        )[:N].real
        rms = np.sqrt(np.mean(mono**2))
        print(f"  Source {k+1}: RMS={rms:.4f}")
        out_path = os.path.join(SEP_DIR, f"mixture_interear_n{n_src}_source_{k+1}.wav")
        save_wav(out_path, mono, sr)

    print()

print("All done.")
