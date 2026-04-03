"""
Deep-learning source separation benchmark.

Models compared
───────────────
All models are out-of-distribution for our task in at least one way —
either trained on fewer speakers, anechoic conditions, or both.
We test them anyway to establish what off-the-shelf DL gives, and to
identify where the gap to ILRMA is and why.

Asteroid models (via HuggingFace):
  convtasnet_libri2mix_clean   2 src, 8 kHz, anechoic  — baseline DL
  convtasnet_libri3mix_noisy   3 src, 8 kHz, noisy     — closer to 4-spk
  dprnn_wham_clean             2 src, 8 kHz, WHAM! (some reverb)

SpeechBrain SepFormer models (stronger architecture — dual-path attention):
  sepformer_whamr              2 src, 8 kHz, WHAM! + reverberation  ← most relevant
  sepformer_libri3mix          3 src, 8 kHz, noisy LibriMix

Key difference from ConvTasNet: SepFormer uses transformer attention over
both local and global context simultaneously. WHAMR training includes
room reverberation, which is the closest match to our recording conditions.

Strategy
────────
Run each model on channel 0 (LF) downsampled to 8 kHz.
Score with DNSMOS ovrl_mos (blind, no reference) and cross-correlation.
Compare against ILRMA baseline (pre-computed in analysis/ilrma/separated/).

Outputs saved to analysis/dl_separation/separated/
"""

import os
import sys
import types
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from speechmos import dnsmos

warnings.filterwarnings("ignore")

# ── Compatibility shim: torchaudio ABI-incompatible with torch 2.9 on Py 3.14.
# Both Asteroid and SpeechBrain only need torchaudio for I/O and losses,
# not for model inference — stubbing it out is safe for our use case.
def _stub_torchaudio():
    fake = types.ModuleType("torchaudio")
    for sub in ["transforms", "functional", "sox_effects", "backend", "pipelines"]:
        m = types.ModuleType(f"torchaudio.{sub}")
        setattr(fake, sub, m)
        sys.modules[f"torchaudio.{sub}"] = m
    sys.modules["torchaudio"] = fake

_stub_torchaudio()

import torch

# Asteroid/SpeechBrain checkpoints pre-date torch 2.6 weights_only=True default.
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from asteroid.models import ConvTasNet, DPRNNTasNet
from speechbrain.inference.separation import SepformerSeparation

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR   = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav"),
}

DNSMOS_SR = 16000


# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: cls, tag/source, loader ("asteroid" or "speechbrain"), n_src, sr, note
MODEL_REGISTRY = {
    # ── Asteroid / ConvTasNet ─────────────────────────────────────────────────
    "convtasnet_libri2mix_clean": {
        "loader": "asteroid",
        "cls": ConvTasNet,
        "tag": "Cosentino/ConvTasNet_LibriMix_sep_clean",
        "n_src": 2, "sr": 8000,
        "note": "ConvTasNet · 2 src · 8 kHz · anechoic LibriMix",
    },
    "convtasnet_libri3mix_noisy": {
        "loader": "asteroid",
        "cls": ConvTasNet,
        "tag": "tmirzaev-dotcom/ConvTasNet_Libri3Mix_sepnoisy",
        "n_src": 3, "sr": 8000,
        "note": "ConvTasNet · 3 src · 8 kHz · noisy LibriMix",
    },
    "dprnn_wham_clean": {
        "loader": "asteroid",
        "cls": DPRNNTasNet,
        "tag": "mpariente/DPRNNTasNet_WHAM!_sepclean",
        "n_src": 2, "sr": 8000,
        "note": "DPRNN · 2 src · 8 kHz · WHAM! (some reverb)",
    },
    # ── SpeechBrain / SepFormer ───────────────────────────────────────────────
    "sepformer_whamr": {
        "loader": "speechbrain",
        "source": "speechbrain/sepformer-whamr",
        "savedir": "/tmp/sepformer-whamr",
        "n_src": 2, "sr": 8000,
        "note": "SepFormer · 2 src · 8 kHz · WHAM!+reverb  ← best domain match",
    },
    "sepformer_libri3mix": {
        "loader": "speechbrain",
        "source": "speechbrain/sepformer-libri3mix",
        "savedir": "/tmp/sepformer-libri3mix",
        "n_src": 3, "sr": 8000,
        "note": "SepFormer · 3 src · 8 kHz · noisy LibriMix",
    },
}


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def save_wav(path, signal, sr):
    peak = np.max(np.abs(signal)) + 1e-12
    out  = np.clip(signal / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"    saved  {os.path.relpath(path)}")


def resample(signal, src_sr, dst_sr):
    gcd = np.gcd(src_sr, dst_sr)
    return resample_poly(signal, dst_sr // gcd, src_sr // gcd)


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_source(sig, native_sr):
    sig_16k = resample(sig, native_sr, DNSMOS_SR).astype(np.float32)
    peak    = np.max(np.abs(sig_16k)) + 1e-12
    return dnsmos.run(sig_16k / peak * 0.9, DNSMOS_SR, return_df=False)["ovrl_mos"]


def cross_corr_score(sources):
    if sources.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(sources)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.sum(np.abs(corr[mask])))


# ── Model loading & inference ─────────────────────────────────────────────────
def load_model(model_key):
    cfg = MODEL_REGISTRY[model_key]
    if cfg["loader"] == "asteroid":
        model = cfg["cls"].from_pretrained(cfg["tag"])
        model.eval()
        return model, "asteroid"
    else:
        model = SepformerSeparation.from_hparams(
            source=cfg["source"],
            savedir=cfg["savedir"],
            run_opts={"device": "cpu"},
        )
        return model, "speechbrain"


def run_model(model_key, mono_8k):
    """mono_8k: 1-D float32 numpy array already at 8 kHz, normalised."""
    model, kind = load_model(model_key)

    if kind == "asteroid":
        x   = torch.from_numpy(mono_8k).unsqueeze(0)   # (1, T)
        with torch.no_grad():
            out = model(x)                              # (1, n_src, T)
        sources = out.squeeze(0).cpu().numpy()          # (n_src, T)

    else:  # speechbrain
        x   = torch.from_numpy(mono_8k).unsqueeze(0)   # (1, T)
        with torch.no_grad():
            out = model.separate_batch(x)               # (1, T, n_src)
        sources = out.squeeze(0).T.cpu().numpy()        # (n_src, T)

    return sources


# ── Main ──────────────────────────────────────────────────────────────────────
def run(wav_key="example"):
    wav_path = WAV_PATHS[wav_key]
    print(f"\nLoading {os.path.relpath(wav_path)} ...")
    sr, data = load_wav(wav_path)
    print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | {data.shape[0]/sr:.1f}s")

    # Prepare 8 kHz mono (channel 0 = LF) for all models
    mono_native = data[:, 0]
    mono_8k = resample(mono_native, sr, 8000).astype(np.float32)
    peak = np.max(np.abs(mono_8k)) + 1e-12
    mono_8k /= peak

    print("\n" + "="*72)
    print("  NOTE: all models below are out-of-distribution for our 4-speaker task.")
    print("  Trained on 2–3 speakers; we have 4. Results show DL baseline capability.")
    print("="*72)

    results = {}
    for model_key, cfg in MODEL_REGISTRY.items():
        print(f"\n  [{model_key}]  {cfg['note']}")
        try:
            sources = run_model(model_key, mono_8k)
            mos_scores = [score_source(sources[k], 8000) for k in range(sources.shape[0])]
            corr = cross_corr_score(sources)

            for k, sig in enumerate(sources):
                out_path = os.path.join(OUT_DIR, f"{wav_key}_{model_key}_src{k+1}.wav")
                save_wav(out_path, sig, 8000)

            results[model_key] = {"mos_scores": mos_scores, "corr": corr}
            print(f"    DNSMOS ovrl_mos: {[f'{m:.3f}' for m in mos_scores]}  "
                  f"mean={np.mean(mos_scores):.3f}  cross-corr={corr:.3f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            results[model_key] = None

    # ── ILRMA baseline ────────────────────────────────────────────────────────
    ilrma_dir   = os.path.join(REPO_ROOT, "analysis", "ilrma", "separated")
    ilrma_files = ["ilrma_0deg_front.wav", "ilrma_90deg_left.wav",
                   "ilrma_180deg_back.wav", "ilrma_270deg_right.wav"]
    ilrma_sources = []
    ilrma_sr_val  = None
    for fn in ilrma_files:
        fp = os.path.join(ilrma_dir, fn)
        if os.path.exists(fp):
            sr_i, sig = wavfile.read(fp)
            if sig.dtype == np.int16:
                sig = sig.astype(np.float64) / 32768.0
            ilrma_sources.append((sig, sr_i))
            ilrma_sr_val = sr_i

    ilrma_mos, ilrma_corr = None, None
    if len(ilrma_sources) == 4:
        ilrma_mos  = [score_source(sig, sr_i) for sig, sr_i in ilrma_sources]
        stacked    = np.stack([sig for sig, _ in ilrma_sources], axis=0)
        ilrma_corr = cross_corr_score(stacked)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  FULL COMPARISON — {wav_key}  |  DNSMOS ovrl_mos (↑) · cross-corr (↓)")
    print("="*72)
    print(f"  {'Method':<32}  {'Srcs':>4}  {'Mean MOS':>9}  {'Corr':>7}  {'Note'}")
    print(f"  {'-'*32}  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*30}")

    for model_key, res in results.items():
        cfg = MODEL_REGISTRY[model_key]
        if res is None:
            print(f"  {model_key:<32}  FAILED")
            continue
        n    = len(res["mos_scores"])
        mean = float(np.mean(res["mos_scores"]))
        print(f"  {model_key:<32}  {n:>4}  {mean:>9.3f}  {res['corr']:>7.3f}  {cfg['note']}")

    if ilrma_mos is not None:
        mean = float(np.mean(ilrma_mos))
        print(f"  {'ILRMA  (classical BSS)':<32}  {4:>4}  {mean:>9.3f}  {ilrma_corr:>7.3f}  "
              f"4 src · 44.1 kHz · uses all 4 channels")
    else:
        print("  [ILRMA outputs not found — run analysis/ilrma/ilrma_separation.py first]")

    print()
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wav", choices=["example", "mixture"], default="example")
    args = p.parse_args()
    run(args.wav)
