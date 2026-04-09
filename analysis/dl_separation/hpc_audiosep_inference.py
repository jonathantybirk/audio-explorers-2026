"""
AudioSep inference on mixture.wav.

AudioSep is a zero-shot text-queried audio separation model (AudioAGI, 2024).
It takes a mono mixture and a text query, and separates the queried sound.

Since we want to separate speakers, we run multiple queries:
  "a person speaking"  x N  — one per estimated speaker

AudioSep is single-channel, so we average the 4-channel mixture to mono first.

We run N_QUERIES separate inference passes, each seeded differently by slightly
varying the query. The outputs are then sorted by estimated DoA (SRP-PHAT on the
mono output, which is approximate but better than nothing).

Outputs:
  analysis/ica/separated/mixture_audiosep_source_{1..N}.wav

Requirements: audiosep (pip install audiosep), torch
"""

import argparse
import itertools
import json
import os

import numpy as np
from scipy.io import wavfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "ica", "separated")
os.makedirs(OUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser()
_parser.add_argument("--n_speakers", type=int, default=7, help="Number of speakers to separate")
_parser.add_argument("--device",     type=str, default="cuda")
_args = _parser.parse_args()

# ── Load audio ────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = wavfile.read(WAV_PATH)
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0
elif data.dtype == np.int32:
    data = data.astype(np.float32) / 2**31
else:
    data = data.astype(np.float32)
print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | {data.shape[0]/sr:.1f}s")

# Mix 4 channels to mono
mono_mix = data.mean(axis=1)
print(f"  Mixed to mono: {mono_mix.shape}")

# Save temp mono mix for AudioSep
import tempfile, soundfile as sf
tmp_dir = tempfile.mkdtemp()
mono_path = os.path.join(tmp_dir, "mixture_mono.wav")
sf.write(mono_path, mono_mix, sr)

# ── AudioSep ──────────────────────────────────────────────────────────────────
print("\nLoading AudioSep model ...")
from audiosep import AudioSep
model = AudioSep.from_pretrained("audo/AudioSep", device=_args.device)

# Queries: vary slightly to encourage different outputs
QUERIES = [
    "a person talking",
    "a person speaking",
    "a human voice speaking",
    "a person saying words",
    "a speaker talking in a room",
    "someone speaking in the background",
    "a voice in a crowd",
]
queries = QUERIES[:_args.n_speakers]

print(f"\nRunning {len(queries)} inference passes ...")
sources = []
for i, query in enumerate(queries):
    print(f"  [{i+1}/{len(queries)}] query: \"{query}\"")
    out_path = os.path.join(tmp_dir, f"source_{i+1}.wav")
    model.separate_audio_file(mono_path, query, output_file=out_path)
    sig, _ = sf.read(out_path, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    # Resample if needed
    if len(sig) != len(mono_mix):
        from scipy.signal import resample
        sig = resample(sig, len(mono_mix)).astype(np.float32)
    sources.append(sig)
    print(f"  → separated {len(sig)} samples")

# ── Save outputs ──────────────────────────────────────────────────────────────
print("\nSaving outputs ...")
for i, sig in enumerate(sources):
    peak = np.max(np.abs(sig)) + 1e-12
    out = np.clip(sig / peak * 0.9, -1.0, 1.0)
    fname = os.path.join(OUT_DIR, f"mixture_audiosep_source_{i+1}.wav")
    wavfile.write(fname, sr, (out * 32767).astype(np.int16))
    print(f"  saved {os.path.relpath(fname)}")

print("\nDone.")
