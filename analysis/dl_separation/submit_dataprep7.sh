#!/bin/sh
### -- LSF job: generate Libri7Mix training data into $BLACKHOLE --
### -- CPU-only. Run this BEFORE submit_sepformer7.sh. --
#BSUB -q hpc
#BSUB -J dataprep7
#BSUB -n 4
#BSUB -W 8:00
#BSUB -R "rusage[mem=24GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o analysis/dl_separation/logs/dataprep7_%J.out
#BSUB -e analysis/dl_separation/logs/dataprep7_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/dl_separation/logs

# Reuse LibriSpeech download from dataprep (Libri4Mix) job if already done
DATA=$BLACKHOLE/libri7mix
LIBRI=$BLACKHOLE/LibriSpeech
mkdir -p "$DATA" "$LIBRI"

echo "=== Downloading LibriSpeech (skips if already present) ==="
cd "$BLACKHOLE"

if [ ! -d "$LIBRI/train-clean-360" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/train-clean-360.tar.gz
    tar -xzf train-clean-360.tar.gz
    rm train-clean-360.tar.gz
fi

if [ ! -d "$LIBRI/dev-clean" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xzf dev-clean.tar.gz
    rm dev-clean.tar.gz
fi

cd -

echo "=== Setting up Python env ==="
module load python3/3.12.11

VENV=$BLACKHOLE/.venv_dataprep
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet soundfile scipy numpy pyroomacoustics

echo "=== Generating Libri7Mix (7-source mixes with room simulation) ==="
python3 - <<'PYEOF'
"""
Generate 7-source reverberant mixes from LibriSpeech.

Each mix: 7 speakers placed at random azimuths in a simulated room,
recorded at 4 microphone positions matching the hearing-aid geometry.

Output layout ($BLACKHOLE/libri7mix/):
  train-360/
    mix/   s1/ s2/ s3/ s4/ s5/ s6/ s7/
  dev/
    mix/   s1/ s2/ s3/ s4/ s5/ s6/ s7/

All files at 16 kHz, 4-second clips, stored as mono WAV for each source
and as a 4-channel WAV for the mix.
"""
import os, glob, random, collections
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import pyroomacoustics as pra

BLACKHOLE = os.environ["BLACKHOLE"]
LIBRI     = os.path.join(BLACKHOLE, "LibriSpeech")
OUT_ROOT  = os.path.join(BLACKHOLE, "libri7mix")
SR_OUT    = 16000
MAX_LEN   = SR_OUT * 4          # 4-second clips
N_SRC     = 7
N_TRAIN   = 20000
N_DEV     =  3000
SEED      = 42
random.seed(SEED)
np.random.seed(SEED)

# Mic geometry matching hearing-aid (4 mics in square ~6 cm apart)
D = 0.065   # inter-ear distance  (m)
L = 0.006   # intra-ear spacing   (m)
MIC_OFFSETS = np.array([
    [ D/2,  L/2, 0.0],
    [ D/2, -L/2, 0.0],
    [-D/2,  L/2, 0.0],
    [-D/2, -L/2, 0.0],
], dtype=np.float64)


def collect_by_speaker(split_dir):
    by_spk = collections.defaultdict(list)
    for f in glob.glob(os.path.join(split_dir, "*", "*", "*.flac")):
        by_spk[os.path.basename(f).split("-")[0]].append(f)
    return dict(by_spk)


def load_resample(path):
    sig, sr = sf.read(path, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != SR_OUT:
        g = np.gcd(sr, SR_OUT)
        sig = resample_poly(sig, SR_OUT // g, sr // g).astype(np.float32)
    return sig


def make_mix_reverb(utterances):
    """Simulate 7 speakers in a random room, record at 4 mics."""
    room_x = random.uniform(4.0, 8.0)
    room_y = random.uniform(4.0, 8.0)
    room_z = random.uniform(2.5, 3.5)
    rt60   = random.uniform(0.15, 0.45)
    e_abs, max_ord = pra.inverse_sabine(rt60, [room_x, room_y, room_z])
    room = pra.ShoeBox(
        [room_x, room_y, room_z],
        fs=SR_OUT,
        materials=pra.Material(e_abs),
        max_order=max_ord,
    )

    # Microphone array centred in room
    cx, cy = room_x / 2, room_y / 2
    mic_pos = (MIC_OFFSETS[:, :2] + np.array([cx, cy])).T   # (2, 4)
    mic_pos_3d = np.vstack([mic_pos, np.full((1, 4), 1.2)])  # add height
    room.add_microphone(mic_pos_3d)

    # Place speakers at 7 random azimuths around centre, 1–2 m away
    azimuths = random.sample(range(0, 360, 10), N_SRC)
    sources_clean = []
    for az in azimuths:
        r   = random.uniform(1.0, 2.0)
        phi = np.deg2rad(az)
        sx  = cx + r * np.sin(phi)
        sy  = cy + r * np.cos(phi)
        sx  = float(np.clip(sx, 0.3, room_x - 0.3))
        sy  = float(np.clip(sy, 0.3, room_y - 0.3))
        sources_clean.append((sx, sy))

    src_signals = []
    for i, path in enumerate(utterances):
        s = load_resample(path)
        if len(s) > MAX_LEN:
            start = random.randint(0, len(s) - MAX_LEN)
            s = s[start: start + MAX_LEN]
        else:
            s = np.pad(s, (0, MAX_LEN - len(s)))
        gain = 10 ** (random.uniform(-3, 3) / 20)
        s = (s * gain).astype(np.float32)
        src_signals.append(s)
        sx, sy = sources_clean[i]
        room.add_source([sx, sy, 1.5], signal=s)

    room.simulate()
    # mic_array.signals: (n_mics, n_samples)
    mix_4ch = room.mic_array.signals.T.astype(np.float32)   # (n_samples, 4)
    # Truncate/pad to MAX_LEN
    if len(mix_4ch) > MAX_LEN:
        mix_4ch = mix_4ch[:MAX_LEN]
    else:
        mix_4ch = np.pad(mix_4ch, ((0, MAX_LEN - len(mix_4ch)), (0, 0)))

    # Normalise mix
    peak = np.max(np.abs(mix_4ch)) + 1e-8
    mix_4ch /= peak
    src_signals = [s / peak for s in src_signals]
    return mix_4ch, src_signals


def generate_split(libri_dir, out_dir, n_mixes):
    os.makedirs(os.path.join(out_dir, "mix"), exist_ok=True)
    for k in range(1, N_SRC + 1):
        os.makedirs(os.path.join(out_dir, f"s{k}"), exist_ok=True)

    by_spk  = collect_by_speaker(libri_dir)
    speakers = list(by_spk.keys())
    assert len(speakers) >= N_SRC, f"Need >= {N_SRC} speakers in {libri_dir}"
    print(f"  {libri_dir}: {len(speakers)} speakers")

    for i in range(n_mixes):
        spk7 = random.sample(speakers, N_SRC)
        utts = [random.choice(by_spk[s]) for s in spk7]
        mix_4ch, sources = make_mix_reverb(utts)

        stem = f"mix_{i:06d}.wav"
        sf.write(os.path.join(out_dir, "mix", stem), mix_4ch, SR_OUT)
        for k, sig in enumerate(sources, 1):
            sf.write(os.path.join(out_dir, f"s{k}", stem),
                     sig.astype(np.float32), SR_OUT)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_mixes} done", flush=True)

    print(f"  Done: {n_mixes} mixes in {out_dir}")


train_out = os.path.join(OUT_ROOT, "train-360")
dev_out   = os.path.join(OUT_ROOT, "dev")
train_done = os.path.isdir(train_out) and \
             len(glob.glob(os.path.join(train_out, "mix", "*.wav"))) >= N_TRAIN
dev_done   = os.path.isdir(dev_out) and \
             len(glob.glob(os.path.join(dev_out,   "mix", "*.wav"))) >= N_DEV

if train_done:
    print("train-360 already complete, skipping.")
else:
    print(f"\nGenerating {N_TRAIN} training mixes...")
    generate_split(os.path.join(LIBRI, "train-clean-360"), train_out, N_TRAIN)

if dev_done:
    print("dev already complete, skipping.")
else:
    print(f"\nGenerating {N_DEV} dev mixes...")
    generate_split(os.path.join(LIBRI, "dev-clean"), dev_out, N_DEV)

print("\n=== Libri7Mix generation complete ===")
PYEOF

echo "=== Done. Libri7Mix written to $DATA ==="
ls "$DATA"
