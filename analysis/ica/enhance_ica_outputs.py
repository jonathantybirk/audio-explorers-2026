"""
Post-process FastMNMF2 / ICA separated sources with:
  1. WPE        — blind dereverberation (nara-wpe)
  2. MetricGAN+ — neural speech enhancement (SpeechBrain)
  3. WPE → MetricGAN+ cascade

Input:  analysis/ica/separated/  (fmnmf2_tuned_source_*.wav, etc.)
Output: analysis/ica/enhanced/   (*_wpe.wav, *_metricgan.wav, *_wpe_metricgan.wav)
"""

import os, glob, warnings
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

warnings.filterwarnings("ignore")

REPO     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_DIR   = os.path.join(REPO, "analysis", "ica", "separated")
OUT_DIR  = os.path.join(REPO, "analysis", "ica", "enhanced")
os.makedirs(OUT_DIR, exist_ok=True)

# Process the best ICA outputs + the mixture versions
PATTERNS = [
    "fmnmf2_tuned_source_*.wav",
    "fmnmf2_tuned_*deg*.wav",
    "mixture_fmnmf2_tuned_source_*.wav",
    "mixture_fmnmf2_tuned_*deg*.wav",
]


def collect_inputs():
    seen = set()
    files = []
    for pat in PATTERNS:
        for f in sorted(glob.glob(os.path.join(IN_DIR, pat))):
            if f not in seen:
                seen.add(f)
                files.append(f)
    return files


def load(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    return sr, data


def save(path, sig, sr):
    peak = np.max(np.abs(sig)) + 1e-12
    out = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved {os.path.relpath(path)}", flush=True)


def resample_to(sig, src_sr, dst_sr):
    if src_sr == dst_sr:
        return sig
    g = gcd(src_sr, dst_sr)
    return resample_poly(sig, dst_sr // g, src_sr // g).astype(np.float32)


# ── WPE ───────────────────────────────────────────────────────────────────────

def apply_wpe(sig, sr):
    """Weighted Prediction Error dereverberation (nara-wpe, single channel)."""
    try:
        from nara_wpe.wpe import wpe_v8 as wpe
        from nara_wpe.utils import stft, istft
    except ImportError:
        print("  nara-wpe not installed, skipping WPE")
        return None

    n_fft = 512
    hop   = 128
    # WPE works per-frequency on STFT; nara-wpe expects (freq, frames) or (channels, freq, frames)
    Y = stft(sig, size=n_fft, shift=hop).T   # (freq, frames)
    Y = Y[np.newaxis]                          # (1, freq, frames)
    dereverbed = wpe(Y, taps=10, delay=3, iterations=5)
    out = istft(dereverbed[0].T, size=n_fft, shift=hop).astype(np.float32)
    # Match length
    if len(out) > len(sig):
        out = out[:len(sig)]
    elif len(out) < len(sig):
        out = np.pad(out, (0, len(sig) - len(out)))
    return out


# ── MetricGAN+ (SpeechBrain) ──────────────────────────────────────────────────

_metricgan_model = None

def get_metricgan():
    global _metricgan_model
    if _metricgan_model is not None:
        return _metricgan_model
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    print("  Loading MetricGAN+ ...", flush=True)
    _metricgan_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="/tmp/metricgan-plus",
        run_opts={"device": "cpu"},
    )
    return _metricgan_model


def apply_metricgan(sig, sr):
    import torch
    model = get_metricgan()
    # MetricGAN+ expects 16 kHz
    sig_16k = resample_to(sig, sr, 16000)
    lengths = torch.tensor([1.0])
    tensor = torch.from_numpy(sig_16k).unsqueeze(0)   # (1, T)
    with torch.no_grad():
        enhanced = model.enhance_batch(tensor, lengths)
    out = enhanced.squeeze().numpy().astype(np.float32)
    out = resample_to(out, 16000, sr)
    if len(out) > len(sig):
        out = out[:len(sig)]
    elif len(out) < len(sig):
        out = np.pad(out, (0, len(sig) - len(out)))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = collect_inputs()
    print(f"Processing {len(files)} files from {IN_DIR}\n")

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{stem}")
        sr, sig = load(path)

        # 1. WPE
        wpe_sig = apply_wpe(sig, sr)
        if wpe_sig is not None:
            save(os.path.join(OUT_DIR, f"{stem}_wpe.wav"), wpe_sig, sr)

        # 2. MetricGAN+ on original
        mg_sig = apply_metricgan(sig, sr)
        if mg_sig is not None:
            save(os.path.join(OUT_DIR, f"{stem}_metricgan.wav"), mg_sig, sr)

        # 3. WPE → MetricGAN+ cascade
        if wpe_sig is not None and mg_sig is not None:
            mg_wpe_sig = apply_metricgan(wpe_sig, sr)
            if mg_wpe_sig is not None:
                save(os.path.join(OUT_DIR, f"{stem}_wpe_metricgan.wav"), mg_wpe_sig, sr)

    print("\nDone.")


if __name__ == "__main__":
    main()
