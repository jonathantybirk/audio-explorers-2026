#!/bin/sh
### -- LSF job: Optuna hyperparameter search for FastMNMF2 n=4 on inter-ear masked mixture --
#BSUB -q hpc
#BSUB -J optuna_n4
#BSUB -n 4
#BSUB -W 8:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs/optuna_n4_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs/optuna_n4_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/doa/logs
mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/ica/separated

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

echo "=== Setting up Python env ==="
module load python3/3.12.11

VENV=.venv_optuna
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet numpy scipy pyroomacoustics optuna

echo "=== Running Optuna n=4 ==="
python3 analysis/doa/optuna_interear_n4.py

echo "=== Done ==="
