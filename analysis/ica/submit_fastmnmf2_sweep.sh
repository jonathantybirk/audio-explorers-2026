#!/bin/sh
### -- LSF job: run FastMNMF2 n_src=5,6,7,8 sweep on mixture.wav (CPU-only) --
#BSUB -q hpc
#BSUB -J fmnmf2_sweep
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o analysis/ica/logs/fmnmf2_sweep_%J.out
#BSUB -e analysis/ica/logs/fmnmf2_sweep_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/ica/logs

module load python3/3.12.11

VENV=.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet pyroomacoustics scipy numpy matplotlib

echo "=== FastMNMF2 n_src sweep (5, 6, 7, 8) on mixture.wav ==="
python3 analysis/ica/fastmnmf2_separation.py --wav mixture

echo "=== Done ==="
