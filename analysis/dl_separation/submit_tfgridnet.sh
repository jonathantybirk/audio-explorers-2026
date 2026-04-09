#!/bin/sh
### -- LSF job script for TF-GridNet 4-source training on DTU HPC --
#BSUB -q gpua100
#BSUB -J tfgridnet4
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/tfgridnet4_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/tfgridnet4_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/dl_separation/logs

module load python3/3.12.11
module load cuda/12.6

VENV=.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch (cu126)..."
    pip install --quiet --upgrade pip
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

if ! python3 -c "import pyroomacoustics" 2>/dev/null; then
    pip install --quiet pyroomacoustics
fi

nvidia-smi

# Smaller model (~8h on A100) — fits comfortably in 24h walltime
python3 analysis/dl_separation/hpc_train_tfgridnet4.py \
    --train-dir $BLACKHOLE/libri4mix/train-360 \
    --val-dir   $BLACKHOLE/libri4mix/dev \
    --epochs 100 \
    --batch-size 16 \
    --num-workers 4 \
    --ckpt-dir analysis/dl_separation/logs/tfgridnet4_ckpt
