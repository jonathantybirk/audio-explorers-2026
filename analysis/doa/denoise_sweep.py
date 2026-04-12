"""
Noise reduction sweep on best convo man clip.
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import wiener, butter, sosfilt

import noisereduce as nr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
IN_PATH   = os.path.join(SEP_DIR, "mixture_interear_n4_source_3.wav")

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    if d.ndim > 1: d = d.mean(axis=1)
    return sr, d.astype(np.float64)

def save_wav(path, sig, sr):
    sig  = np.nan_to_num(sig)
    peak = np.max(np.abs(sig)) + 1e-9
    out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")

sr, data = load_wav(IN_PATH)
data = data.astype(np.float32)

results = {}

# 1. noisereduce non-stationary prop=0.8 (existing baseline)
results["nr_nonstat_08"] = nr.reduce_noise(y=data, sr=sr, stationary=False, prop_decrease=0.8)

# 2. noisereduce non-stationary prop=0.9
results["nr_nonstat_09"] = nr.reduce_noise(y=data, sr=sr, stationary=False, prop_decrease=0.9)

# 3. noisereduce non-stationary prop=1.0
results["nr_nonstat_10"] = nr.reduce_noise(y=data, sr=sr, stationary=False, prop_decrease=1.0)

# 4. noisereduce stationary prop=0.8
results["nr_stat_08"] = nr.reduce_noise(y=data, sr=sr, stationary=True, prop_decrease=0.8)

# 5. noisereduce stationary prop=1.0
results["nr_stat_10"] = nr.reduce_noise(y=data, sr=sr, stationary=True, prop_decrease=1.0)

# 6. noisereduce non-stationary aggressive threshold
results["nr_aggressive"] = nr.reduce_noise(y=data, sr=sr, stationary=False,
                                            prop_decrease=1.0,
                                            thresh_n_mult_nonstationary=3.0)

# 7. scipy Wiener filter (default kernel 3)
results["wiener_3"] = wiener(data, mysize=3).astype(np.float32)

# 8. scipy Wiener filter (kernel 7)
results["wiener_7"] = wiener(data, mysize=7).astype(np.float32)

# 9. bandpass 100Hz-8kHz then noisereduce
sos = butter(6, [100, 8000], btype='bandpass', fs=sr, output='sos')
bp  = sosfilt(sos, data).astype(np.float32)
results["bandpass_then_nr"] = nr.reduce_noise(y=bp, sr=sr, stationary=False, prop_decrease=0.9)

# 10. bandpass only
results["bandpass_only"] = bp

print("Saving ...")
for name, sig in results.items():
    save_wav(os.path.join(SEP_DIR, f"convoman_denoise_{name}.wav"), sig, sr)

print("\nDone.")
