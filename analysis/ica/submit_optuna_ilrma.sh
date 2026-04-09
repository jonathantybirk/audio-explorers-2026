#!/bin/sh
### -- LSF job: Optuna hyperparameter search for ILRMA on mixture.wav --
### -- CPU-only job, 80 trials. --
#BSUB -q hpc
#BSUB -J optuna_ilrma
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o analysis/ica/optuna/logs/optuna_ilrma_%J.out
#BSUB -e analysis/ica/optuna/logs/optuna_ilrma_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/ica/optuna/logs

module load python3/3.12.11

VENV=.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet pyroomacoustics optuna scipy numpy

echo "=== ILRMA Optuna search ==="
python3 analysis/ica/hpc_optuna_ilrma.py \
    --n_trials 80 \
    --n_jobs   1

echo "=== Done ==="
