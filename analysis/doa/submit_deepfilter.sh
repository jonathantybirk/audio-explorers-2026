#!/bin/sh
### -- LSF job: DeepFilterNet inference on convo man isolation --
#BSUB -q hpc
#BSUB -J deepfilter
#BSUB -n 4
#BSUB -W 1:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs/deepfilter_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs/deepfilter_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

module load python3/3.12.11

VENV=.venv_deepfilter
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet deepfilternet scipy numpy

echo "=== Running DeepFilterNet inference ==="
python3 - <<'PYEOF'
import os
import numpy as np
from scipy.io import wavfile
from df.enhance import enhance, init_df, load_audio, save_audio

REPO_ROOT = os.path.abspath(".")
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")

model, df_state, _ = init_df()

inputs = [
    ("mixture_interear_n4_source_3.wav",  "convoman_deepfilter_raw.wav"),
    ("convoman_noisereduced.wav",          "convoman_deepfilter_nr_then_df.wav"),
]

for in_name, out_name in inputs:
    in_path  = os.path.join(SEP_DIR, in_name)
    out_path = os.path.join(SEP_DIR, out_name)
    if not os.path.exists(in_path):
        print(f"  SKIP (not found): {in_name}")
        continue
    audio, _ = load_audio(in_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(out_path, enhanced, df_state.sr())
    print(f"  saved {out_name}")

print("Done.")
PYEOF

echo "=== Done ==="
