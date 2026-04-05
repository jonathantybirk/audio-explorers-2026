#!/bin/sh
### -- LSF job script for SepFormer 3→4 LoRA fine-tune on DTU HPC --
### -- specify queue -- try A100 first; switch to gpuv100 if queue is long
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J sepformer4
### -- ask for number of cores --
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- max 24h on GPU queues --
#BSUB -W 24:00
### -- request system memory --
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o analysis/dl_separation/logs/sepformer4_%J.out
#BSUB -e analysis/dl_separation/logs/sepformer4_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/dl_separation/logs

module load python3/3.12.11
module load cuda/12.6

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

if ! python3 -c "import torch" 2>/dev/null; then
    echo "First run — installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

if ! python3 -c "import speechbrain" 2>/dev/null; then
    pip install --quiet speechbrain
fi

if ! python3 -c "import pyroomacoustics" 2>/dev/null; then
    pip install --quiet pyroomacoustics
fi

if ! python3 -c "import peft" 2>/dev/null; then
    echo "Installing peft for LoRA..."
    pip install --quiet peft
fi

nvidia-smi

# H100 80GB → batch 24; A100 80GB → batch 24; A100 40GB → batch 16; V100 32GB → batch 12
python3 analysis/dl_separation/hpc_finetune_sepformer4.py \
    --train-dir /work3/s216136/libri4mix/train-360 \
    --val-dir   /work3/s216136/libri4mix/dev \
    --epochs 30 \
    --batch-size 24 \
    --num-workers 4 \
    --ckpt-dir analysis/dl_separation/logs/sepformer4_ckpt
