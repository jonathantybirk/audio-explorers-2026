#!/bin/sh
### -- LSF job script: download LibriSpeech + generate Libri4Mix into $BLACKHOLE --
### -- Runs on a CPU node, no GPU needed. Submit this BEFORE the training jobs. --
#BSUB -q hpc
#BSUB -J dataprep
#BSUB -n 4
#BSUB -W 6:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o analysis/dl_separation/logs/dataprep_%J.out
#BSUB -e analysis/dl_separation/logs/dataprep_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/dl_separation/logs

DATA=$BLACKHOLE/libri4mix
LIBRI=$BLACKHOLE/LibriSpeech
mkdir -p "$DATA" "$LIBRI"

# ── 1. Download LibriSpeech ────────────────────────────────────────────────────
echo "=== Downloading LibriSpeech ==="
cd "$BLACKHOLE"

if [ ! -d "$LIBRI/train-clean-360" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/train-clean-360.tar.gz
    tar -xzf train-clean-360.tar.gz   # extracts to LibriSpeech/train-clean-360/
    rm train-clean-360.tar.gz
fi

if [ ! -d "$LIBRI/dev-clean" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xzf dev-clean.tar.gz         # extracts to LibriSpeech/dev-clean/
    rm dev-clean.tar.gz
fi

# ── 2. Generate 4-source mixes ────────────────────────────────────────────────
echo "=== Setting up Python env ==="
module load python3/3.12.11

VENV=$BLACKHOLE/.venv_dataprep
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet soundfile scipy numpy

echo "=== Generating Libri4Mix (self-contained, no LibriMix metadata required) ==="
python3 - <<'PYEOF'
"""
Generate 4-source mixes from LibriSpeech without the LibriMix toolkit.
Output layout (matches what hpc_finetune_sepformer4.py and hpc_train_tfgridnet4.py expect):

  $BLACKHOLE/libri4mix/
    train-360/
      mix_clean/  s1/  s2/  s3/  s4/
    dev/
      mix_clean/  s1/  s2/  s3/  s4/
"""
import os, glob, random, collections
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BLACKHOLE  = os.environ["BLACKHOLE"]
LIBRI      = os.path.join(BLACKHOLE, "LibriSpeech")
OUT_ROOT   = os.path.join(BLACKHOLE, "libri4mix")
SR_OUT     = 8000
MAX_LEN_S  = 4.0
MAX_LEN    = int(MAX_LEN_S * SR_OUT)
SEED       = 42

N_TRAIN    = 20000   # number of 4-speaker mixes for training
N_DEV      =  3000   # number of 4-speaker mixes for validation

random.seed(SEED)
np.random.seed(SEED)

def collect_by_speaker(libri_split_dir):
    """Return dict: speaker_id -> [flac_path, ...]"""
    spk = collections.defaultdict(list)
    for f in glob.glob(os.path.join(libri_split_dir, "*", "*", "*.flac")):
        speaker_id = os.path.basename(f).split("-")[0]
        spk[speaker_id].append(f)
    return dict(spk)

def load_resample(path, sr_out):
    sig, sr = sf.read(path, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != sr_out:
        g = np.gcd(sr, sr_out)
        sig = resample_poly(sig, sr_out // g, sr // g).astype(np.float32)
    return sig

def make_mix(utterances, max_len):
    """Load, truncate/pad, random-scale and sum 4 utterances."""
    sources = []
    for path in utterances:
        s = load_resample(path, SR_OUT)
        if len(s) > max_len:
            start = random.randint(0, len(s) - max_len)
            s = s[start: start + max_len]
        else:
            s = np.pad(s, (0, max_len - len(s)))
        # Random gain ±3 dB
        gain = 10 ** (random.uniform(-3, 3) / 20)
        sources.append(s * gain)
    mix = np.sum(sources, axis=0).astype(np.float32)
    peak = np.max(np.abs(mix)) + 1e-8
    sources = [s / peak for s in sources]
    mix = mix / peak
    return mix, sources

def generate_split(libri_dir, out_dir, n_mixes):
    os.makedirs(os.path.join(out_dir, "mix_clean"), exist_ok=True)
    for k in range(1, 5):
        os.makedirs(os.path.join(out_dir, f"s{k}"), exist_ok=True)

    by_spk = collect_by_speaker(libri_dir)
    speakers = list(by_spk.keys())
    assert len(speakers) >= 4, f"Need >= 4 speakers, found {len(speakers)} in {libri_dir}"
    print(f"  {libri_dir}: {len(speakers)} speakers, {sum(len(v) for v in by_spk.values())} utterances")

    for i in range(n_mixes):
        spk4 = random.sample(speakers, 4)
        utts = [random.choice(by_spk[s]) for s in spk4]
        mix, sources = make_mix(utts, MAX_LEN)

        stem = f"mix_{i:06d}.wav"
        sf.write(os.path.join(out_dir, "mix_clean", stem), mix, SR_OUT)
        for k, sig in enumerate(sources, 1):
            sf.write(os.path.join(out_dir, f"s{k}", stem), sig, SR_OUT)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_mixes} mixes written", flush=True)

    print(f"  Done: {n_mixes} mixes in {out_dir}")

train_out = os.path.join(OUT_ROOT, "train-360")
dev_out   = os.path.join(OUT_ROOT, "dev")

if os.path.isdir(train_out) and len(glob.glob(os.path.join(train_out, "mix_clean", "*.wav"))) >= N_TRAIN:
    print(f"train-360 already exists ({N_TRAIN} mixes), skipping.")
else:
    print(f"\nGenerating {N_TRAIN} training mixes...")
    generate_split(os.path.join(LIBRI, "train-clean-360"), train_out, N_TRAIN)

if os.path.isdir(dev_out) and len(glob.glob(os.path.join(dev_out, "mix_clean", "*.wav"))) >= N_DEV:
    print(f"dev already exists ({N_DEV} mixes), skipping.")
else:
    print(f"\nGenerating {N_DEV} dev mixes...")
    generate_split(os.path.join(LIBRI, "dev-clean"), dev_out, N_DEV)

print("\n=== Libri4Mix generation complete ===")
print(f"train-360: {OUT_ROOT}/train-360/")
print(f"dev:       {OUT_ROOT}/dev/")
PYEOF

echo "=== Done. Libri4Mix written to $DATA ==="
ls "$DATA"
