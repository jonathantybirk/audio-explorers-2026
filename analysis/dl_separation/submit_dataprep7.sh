#!/bin/sh
### -- LSF job: generate Libri7Mix training data into $BLACKHOLE --
### -- CPU-only. Run this BEFORE submit_sepformer7.sh. --
#BSUB -q hpc
#BSUB -J dataprep7
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=24GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/dataprep7_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/dataprep7_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

if [ -z "$BLACKHOLE" ] || [ ! -d "$BLACKHOLE" ]; then
    echo "ERROR: \$BLACKHOLE not set or not mounted on this node." >&2; exit 1
fi

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

VENV=.venv_dataprep
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

Parallelised with multiprocessing.Pool (N_WORKERS=4) and max_order capped
at 3 to keep each room simulation fast (~2-5 s/mix vs ~50 s previously).
Resumable: skips mixes whose output files already exist.
"""
import os, glob, random, collections, multiprocessing
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import pyroomacoustics as pra

BLACKHOLE  = os.environ["BLACKHOLE"]
LIBRI      = os.path.join(BLACKHOLE, "LibriSpeech")
OUT_ROOT   = os.path.join(BLACKHOLE, "libri7mix")
SR_OUT     = 16000
MAX_LEN    = SR_OUT * 4   # 4-second clips
N_SRC      = 7
N_TRAIN    = 20000
N_DEV      = 3000
N_WORKERS  = 4
MAX_ORDER  = 3            # cap reflection order — uncapped was the bottleneck

D = 0.065
L = 0.006
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


def load_resample(path, sr_out):
    sig, sr = sf.read(path, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != sr_out:
        g = np.gcd(sr, sr_out)
        sig = resample_poly(sig, sr_out // g, sr // g).astype(np.float32)
    return sig


def make_one_mix(args):
    """Worker function — must be top-level for multiprocessing pickling."""
    idx, utterances, out_dir, seed = args
    stem = f"mix_{idx:06d}.wav"
    mix_path = os.path.join(out_dir, "mix", stem)
    if os.path.exists(mix_path):
        return idx, "skip"   # already done — resume

    try:
        rng = random.Random(seed)

        room_x = rng.uniform(4.0, 8.0)
        room_y = rng.uniform(4.0, 8.0)
        room_z = rng.uniform(2.5, 3.5)
        rt60   = rng.uniform(0.15, 0.45)
        e_abs, max_ord = pra.inverse_sabine(rt60, [room_x, room_y, room_z])
        room = pra.ShoeBox(
            [room_x, room_y, room_z],
            fs=SR_OUT,
            materials=pra.Material(e_abs),
            max_order=min(max_ord, MAX_ORDER),   # cap for speed
        )

        cx, cy = room_x / 2, room_y / 2
        mic_pos = (MIC_OFFSETS[:, :2] + np.array([cx, cy])).T
        mic_pos_3d = np.vstack([mic_pos, np.full((1, 4), 1.2)])
        room.add_microphone(mic_pos_3d)

        azimuths = rng.sample(range(0, 360, 10), N_SRC)
        src_xy = []
        for az in azimuths:
            r   = rng.uniform(1.0, 2.0)
            phi = np.deg2rad(az)
            sx  = float(np.clip(cx + r * np.sin(phi), 0.3, room_x - 0.3))
            sy  = float(np.clip(cy + r * np.cos(phi), 0.3, room_y - 0.3))
            src_xy.append((sx, sy))

        src_signals = []
        for i, path in enumerate(utterances):
            s = load_resample(path, SR_OUT)   # may raise on corrupted FLAC
            if len(s) > MAX_LEN:
                start = rng.randint(0, len(s) - MAX_LEN)
                s = s[start: start + MAX_LEN]
            else:
                s = np.pad(s, (0, MAX_LEN - len(s)))
            gain = 10 ** (rng.uniform(-3, 3) / 20)
            s = (s * gain).astype(np.float32)
            src_signals.append(s)
            sx, sy = src_xy[i]
            room.add_source([sx, sy, 1.5], signal=s)

        room.simulate()
        mix_4ch = room.mic_array.signals.T.astype(np.float32)
        if len(mix_4ch) > MAX_LEN:
            mix_4ch = mix_4ch[:MAX_LEN]
        else:
            mix_4ch = np.pad(mix_4ch, ((0, MAX_LEN - len(mix_4ch)), (0, 0)))

        peak = np.max(np.abs(mix_4ch)) + 1e-8
        mix_4ch    /= peak
        src_signals = [s / peak for s in src_signals]

        sf.write(mix_path, mix_4ch, SR_OUT)
        for k, sig in enumerate(src_signals, 1):
            sf.write(os.path.join(out_dir, f"s{k}", stem), sig, SR_OUT)

        return idx, "ok"

    except Exception as e:
        # Corrupted FLAC or simulation error — log and skip, don't crash the pool
        return idx, f"error: {e}"


def generate_split(libri_dir, out_dir, n_mixes, base_seed):
    os.makedirs(os.path.join(out_dir, "mix"), exist_ok=True)
    for k in range(1, N_SRC + 1):
        os.makedirs(os.path.join(out_dir, f"s{k}"), exist_ok=True)

    by_spk   = collect_by_speaker(libri_dir)
    speakers = sorted(by_spk.keys())
    assert len(speakers) >= N_SRC, f"Need >= {N_SRC} speakers in {libri_dir}"
    print(f"  {libri_dir}: {len(speakers)} speakers", flush=True)

    # Build deterministic utterance lists per mix (reproducible across restarts)
    master_rng = random.Random(base_seed)
    tasks = []
    for i in range(n_mixes):
        spk7 = master_rng.sample(speakers, N_SRC)
        utts = [master_rng.choice(by_spk[s]) for s in spk7]
        tasks.append((i, utts, out_dir, base_seed + i))

    done = skipped = errors = 0
    with multiprocessing.Pool(N_WORKERS) as pool:
        for idx, status in pool.imap_unordered(make_one_mix, tasks, chunksize=8):
            if status == "skip":
                skipped += 1
            elif status == "ok":
                done += 1
            else:
                errors += 1
                print(f"  WARNING mix_{idx:06d}: {status}", flush=True)
            total = done + skipped + errors
            if total % 500 == 0:
                print(f"  {total}/{n_mixes}  new={done} skipped={skipped} errors={errors}", flush=True)

    print(f"  Done: {out_dir}  new={done} skipped={skipped} errors={errors}")


train_out  = os.path.join(OUT_ROOT, "train-360")
dev_out    = os.path.join(OUT_ROOT, "dev")
train_done = os.path.isdir(train_out) and \
             len(glob.glob(os.path.join(train_out, "mix", "*.wav"))) >= N_TRAIN
dev_done   = os.path.isdir(dev_out) and \
             len(glob.glob(os.path.join(dev_out,   "mix", "*.wav"))) >= N_DEV

if train_done:
    print("train-360 already complete, skipping.")
else:
    existing = len(glob.glob(os.path.join(train_out, "mix", "*.wav")))
    print(f"\nGenerating {N_TRAIN} training mixes ({existing} already exist) ...")
    generate_split(os.path.join(LIBRI, "train-clean-360"), train_out, N_TRAIN, base_seed=42)

if dev_done:
    print("dev already complete, skipping.")
else:
    existing = len(glob.glob(os.path.join(dev_out, "mix", "*.wav")))
    print(f"\nGenerating {N_DEV} dev mixes ({existing} already exist) ...")
    generate_split(os.path.join(LIBRI, "dev-clean"), dev_out, N_DEV, base_seed=9999)

print("\n=== Libri7Mix generation complete ===")
PYEOF

echo "=== Done. Libri7Mix written to $DATA ==="
ls "$DATA"
