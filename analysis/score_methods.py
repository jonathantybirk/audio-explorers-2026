"""
Blind quality scoring across all separation methods.

Metrics (no reference signal needed):
  OVRL  — DNSMOS P.835 overall MOS (1–5, higher = better)
  SIG   — DNSMOS speech quality score
  BAK   — DNSMOS background noise score
  SFLATNESS — spectral flatness of output (lower = more tonal/speech-like)
  RMS_DB    — output level in dBFS

Usage:
  python analysis/score_methods.py
"""

import os, glob, warnings
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from speechmos import dnsmos

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEARCH_DIRS = [
    os.path.join(REPO, "analysis", "ica", "separated"),
    os.path.join(REPO, "analysis", "dl_separation", "separated"),
]

TARGET_SR = 16000   # DNSMOS expects 16 kHz


def load_mono_16k(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample to 16 kHz if needed
    if sr != TARGET_SR:
        from math import gcd
        g = gcd(sr, TARGET_SR)
        from scipy.signal import resample_poly
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    return data


def spectral_flatness(sig, sr=TARGET_SR, nperseg=512):
    _, _, Sxx = spectrogram(sig, fs=sr, nperseg=nperseg)
    power = Sxx.mean(axis=1) + 1e-12
    geom_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    return float(geom_mean / arith_mean)


def score_file(path):
    sig = load_mono_16k(path)
    # DNSMOS
    result = dnsmos.run(sig, TARGET_SR, return_df=False)
    ovrl = float(result["ovrl_mos"])
    sig_mos = float(result["sig_mos"])
    bak = float(result["bak_mos"])
    # Spectral flatness
    sf = spectral_flatness(sig)
    # RMS
    rms = float(20 * np.log10(np.sqrt(np.mean(sig**2)) + 1e-12))
    return {"OVRL": ovrl, "SIG": sig_mos, "BAK": bak, "SFLATNESS": sf, "RMS_dB": rms}


def shorten_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    # Split into split (example/mixture) and method
    for prefix in ("example_", "mixture_"):
        if name.startswith(prefix):
            return prefix.replace("_", ""), name[len(prefix):]
    return "?", name


def collect_files():
    files = {}
    for d in SEARCH_DIRS:
        for f in glob.glob(os.path.join(d, "*.wav")):
            split, key = shorten_name(f)
            if split not in files:
                files[split] = {}
            files[split][key] = f
    return files


def main():
    all_files = collect_files()

    for split in ("example", "mixture"):
        if split not in all_files:
            print(f"\n=== {split.upper()} — no files found ===\n")
            continue

        print(f"\n{'='*80}")
        print(f"  {split.upper()} MIXTURE — blind quality scores")
        print(f"{'='*80}")
        print(f"  {'Method':<55} {'OVRL':>6} {'SIG':>6} {'BAK':>6} {'SFLAT':>7} {'RMS':>7}")
        print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")

        rows = []
        for key, path in sorted(all_files[split].items()):
            try:
                scores = score_file(path)
                rows.append((key, scores))
            except Exception as e:
                print(f"  {'ERR: '+key:<55}  {str(e)}")

        # Sort by OVRL descending
        rows.sort(key=lambda x: x[1]["OVRL"], reverse=True)
        for key, s in rows:
            flag = " ★" if s["OVRL"] >= 3.5 else ""
            print(
                f"  {key:<55} {s['OVRL']:>6.2f} {s['SIG']:>6.2f} {s['BAK']:>6.2f}"
                f" {s['SFLATNESS']:>7.4f} {s['RMS_dB']:>6.1f}{flag}"
            )

    print()


if __name__ == "__main__":
    main()
