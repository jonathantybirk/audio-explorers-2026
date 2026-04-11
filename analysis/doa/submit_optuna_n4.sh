#!/bin/sh
### -- LSF job: Optuna hyperparameter search for FastMNMF2 n=4 on inter-ear masked mixture --
#BSUB -q hpc
#BSUB -J optuna_n4
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs/optuna_n4_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs/optuna_n4_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs
mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/ica/separated

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

if [ ! -f "DONT-TOUCH/Software Case/mixture.wav" ]; then
    echo "ERROR: mixture.wav not found." >&2; exit 1
fi

module load python3/3.12.11

VENV=.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet pyroomacoustics optuna scipy numpy

echo "=== Generating inter-ear masked mixture ==="
python3 analysis/doa/isolate_convoman4.py

echo "=== Running Optuna n=4 ==="
python3 analysis/doa/optuna_interear_n4.py

echo "=== Done ==="
