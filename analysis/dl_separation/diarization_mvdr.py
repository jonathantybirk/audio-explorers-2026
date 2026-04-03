"""
Diarization-guided MVDR beamforming using pyannote.audio.

Approach
────────
Instead of blind BSS (like ILRMA), we:
  1. Run pyannote speaker diarization to get per-speaker activity masks
     (who is speaking at each time frame).
  2. For each speaker: estimate a "target" covariance matrix from frames
     where only that speaker is active, and an "interference" covariance
     from frames where that speaker is silent.
  3. Compute MVDR beamforming weights from these covariances and apply to
     the 4-channel mixture.

This is the production-style approach used in hearing aids and meeting
transcription systems: DL tells us *who spoke when*, classical spatial
filtering does the actual separation.

The key advantage over blind ILRMA: we can leverage the DL model's
understanding of speech/non-speech and speaker identity, while still
exploiting the full spatial information from all 4 microphones.

Setup required (one-time)
──────────────────────────
pyannote models are gated on HuggingFace. You need to:
  1. Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Accept conditions at https://huggingface.co/pyannote/segmentation-3.0
  3. Create a HF token at https://huggingface.co/settings/tokens
  4. Set environment variable: export HF_TOKEN=hf_...
     (add to ~/.zshrc to make permanent)

Usage
─────
  export HF_TOKEN=hf_your_token_here
  python analysis/dl_separation/diarization_mvdr.py --wav example
  python analysis/dl_separation/diarization_mvdr.py --wav mixture

Outputs saved to analysis/dl_separation/separated/diar_mvdr_*.wav
"""

import argparse
import json
import os
import sys
import types
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")

# ── torchaudio shim (ABI-incompatible with torch 2.9 on Python 3.14) ─────────
def _stub_torchaudio():
    fake = types.ModuleType("torchaudio")
    for sub in ["transforms", "functional", "sox_effects", "backend", "pipelines"]:
        m = types.ModuleType(f"torchaudio.{sub}")
        setattr(fake, sub, m)
        sys.modules[f"torchaudio.{sub}"] = m
    sys.modules["torchaudio"] = fake

_stub_torchaudio()

import torch
_orig_load = torch.load
def _pl(f, *a, **kw): kw.setdefault("weights_only", False); return _orig_load(f, *a, **kw)
torch.load = _pl

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

WAV_PATHS = {
    "example": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "example_mixture.wav"),
    "mixture": os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav"),
}

# ── STFT / MVDR parameters ────────────────────────────────────────────────────
STFT_SIZE    = 2048
HOP_SIZE     = 512
DIAG_LOADING = 1e-4   # regularisation for covariance inversion


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


# ── STFT helpers ──────────────────────────────────────────────────────────────
def stft_multichannel(data, n_fft, hop):
    """data: (N, C). Returns (F, T, C) complex."""
    n_ch = data.shape[1]
    win  = np.hanning(n_fft)
    n_frames = 1 + (data.shape[0] - n_fft) // hop
    out = np.zeros((n_fft // 2 + 1, n_frames, n_ch), dtype=np.complex128)
    for c in range(n_ch):
        for t in range(n_frames):
            frame = data[t * hop: t * hop + n_fft, c] * win
            out[:, t, c] = np.fft.rfft(frame)
    return out


def istft_mono(X, hop, n_samples):
    """X: (F, T) complex. Returns (N,) real."""
    n_fft   = (X.shape[0] - 1) * 2
    win     = np.hanning(n_fft)
    n_frames = X.shape[1]
    out     = np.zeros(n_fft + hop * (n_frames - 1))
    norm    = np.zeros_like(out)
    for t in range(n_frames):
        frame = np.fft.irfft(X[:, t], n=n_fft) * win
        out[t * hop: t * hop + n_fft]  += frame
        norm[t * hop: t * hop + n_fft] += win ** 2
    norm   = np.where(norm > 1e-8, norm, 1.0)
    return (out / norm)[:n_samples]


# ── MVDR beamforming ──────────────────────────────────────────────────────────
def compute_covariance(X_freq, mask):
    """
    X_freq: (T, C) complex — one frequency bin, all frames and channels.
    mask:   (T,)  float in [0, 1] — soft mask indicating target activity.
    Returns (C, C) complex covariance matrix.
    """
    T, C   = X_freq.shape
    w      = mask / (mask.sum() + 1e-8)
    Sigma  = np.zeros((C, C), dtype=np.complex128)
    for t in range(T):
        x = X_freq[t, :, np.newaxis]   # (C, 1)
        Sigma += w[t] * (x @ x.conj().T)
    return Sigma


def mvdr_weights(Rtarget, Rnoise, ref_ch=0):
    """
    Compute MVDR beamforming weights for one frequency bin.
    Rtarget, Rnoise: (C, C) complex covariance matrices.
    Returns (C,) complex weight vector.
    """
    C      = Rtarget.shape[0]
    Rn_reg = Rnoise + DIAG_LOADING * np.eye(C)
    try:
        Rn_inv = np.linalg.inv(Rn_reg)
    except np.linalg.LinAlgError:
        return np.zeros(C, dtype=np.complex128)

    # Steering vector from target covariance (generalised eigenvector approach)
    try:
        eigvals, eigvecs = np.linalg.eigh(Rtarget)
        rtf = eigvecs[:, -1]           # principal eigenvector
    except np.linalg.LinAlgError:
        rtf = np.ones(C, dtype=np.complex128) / np.sqrt(C)

    num = Rn_inv @ rtf
    den = rtf.conj() @ num + 1e-12
    w   = num / den
    # Scale so reference channel has unit gain (distortionless response)
    w  /= (rtf.conj() @ w + 1e-12)
    return w


def apply_mvdr(X, target_mask, n_samples):
    """
    X:           (F, T, C) complex STFT of the 4-channel mixture.
    target_mask: (T,) soft mask — 1 where target speaker is active.
    Returns mono separated waveform (n_samples,).
    """
    F, T, C = X.shape
    noise_mask = 1.0 - target_mask

    Y = np.zeros((F, T), dtype=np.complex128)
    for f in range(F):
        Xf       = X[f, :, :]                          # (T, C)
        Rt       = compute_covariance(Xf, target_mask)
        Rn       = compute_covariance(Xf, noise_mask)
        w        = mvdr_weights(Rt, Rn)                # (C,)
        Y[f, :]  = Xf @ w.conj()                       # (T,)

    return istft_mono(Y, HOP_SIZE, n_samples)


# ── Pyannote diarization → frame-level masks ──────────────────────────────────
def diarize(wav_path, hf_token):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    return pipeline(wav_path)


def diarization_to_masks(diarization, n_frames, sr, hop):
    """Convert pyannote Annotation to per-speaker frame-level soft masks."""
    speakers = sorted(diarization.labels())
    masks    = {spk: np.zeros(n_frames, dtype=np.float32) for spk in speakers}

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        t_start = int(segment.start * sr / hop)
        t_end   = min(int(segment.end   * sr / hop) + 1, n_frames)
        masks[speaker][t_start:t_end] = 1.0

    return masks


# ── Main ──────────────────────────────────────────────────────────────────────
def run(wav_key, hf_token):
    wav_path = WAV_PATHS[wav_key]
    print(f"\nLoading {os.path.relpath(wav_path)} ...")
    sr, data = load_wav(wav_path)
    n_samples = data.shape[0]
    print(f"  {n_samples} samples | {data.shape[1]} ch | {sr} Hz | {n_samples/sr:.1f}s")

    print("\nRunning pyannote speaker diarization ...")
    diarization = diarize(wav_path, hf_token)
    speakers    = sorted(diarization.labels())
    print(f"  Found {len(speakers)} speakers: {speakers}")
    for segment, _, spk in diarization.itertracks(yield_label=True):
        print(f"    {spk}  {segment.start:.2f}s – {segment.end:.2f}s")

    print("\nComputing STFT ...")
    X = stft_multichannel(data, STFT_SIZE, HOP_SIZE)   # (F, T, C)
    F, T, C = X.shape
    print(f"  Shape: {F} bins × {T} frames × {C} channels")

    masks = diarization_to_masks(diarization, T, sr, HOP_SIZE)

    print("\nApplying MVDR beamforming per speaker ...")
    from speechmos import dnsmos

    def score(sig):
        from scipy.signal import resample_poly as rsp
        sig16 = rsp(sig, 16000, sr).astype(np.float32)
        peak  = np.max(np.abs(sig16)) + 1e-12
        return dnsmos.run(sig16 / peak * 0.9, 16000, return_df=False)["ovrl_mos"]

    mos_scores = []
    for spk in speakers:
        mask = masks[spk]
        print(f"  Speaker {spk}: activity {mask.mean()*100:.1f}% of frames")
        separated = apply_mvdr(X, mask, n_samples)
        mos       = score(separated)
        mos_scores.append(mos)
        out_path  = os.path.join(OUT_DIR, f"{wav_key}_diar_mvdr_{spk}.wav")
        save_wav(out_path, separated, sr)
        print(f"    DNSMOS ovrl_mos: {mos:.3f}")

    print(f"\n  Mean DNSMOS: {np.mean(mos_scores):.3f}")
    print("  (Compare against ILRMA baseline — run analysis/ilrma/ilrma_separation.py)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Diarization-guided MVDR beamforming")
    p.add_argument("--wav", choices=["example", "mixture"], default="example")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    args = p.parse_args()

    if not args.hf_token:
        print("""
ERROR: HuggingFace token required for pyannote models.

Setup (one-time):
  1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
     and accept the user conditions (requires HF account).
  2. Also accept: https://huggingface.co/pyannote/segmentation-3.0
  3. Get a token at https://huggingface.co/settings/tokens
  4. Run: export HF_TOKEN=hf_your_token_here
     (add to ~/.zshrc to persist)
  5. Re-run this script.
""")
        sys.exit(1)

    run(args.wav, args.hf_token)
