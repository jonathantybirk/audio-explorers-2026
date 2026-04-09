#!/bin/sh
### -- LSF job: Optuna hyperparameter search for FastMNMF2 on mixture.wav --
### -- CPU-only job. Searches n_src=5,6,7,8 with 50 trials each. --
#BSUB -q hpc
#BSUB -J optuna_fmnmf2
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/ica/optuna/logs/optuna_fmnmf2_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/ica/optuna/logs/optuna_fmnmf2_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/ica/optuna/logs

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

# Fail fast: verify input file exists
if [ ! -f "DONT-TOUCH/Software Case/mixture.wav" ]; then
    echo "ERROR: mixture.wav not found. Wrong working directory?" >&2; exit 1
fi

module load python3/3.12.11

VENV=.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet pyroomacoustics optuna scipy numpy tqdm

echo "=== FastMNMF2 Optuna search ==="
python3 analysis/ica/hpc_optuna_fastmnmf2.py \
    --n_trials 50 \
    --n_jobs   1

echo "=== Done ==="
