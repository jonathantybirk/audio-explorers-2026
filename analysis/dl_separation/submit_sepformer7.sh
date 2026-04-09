#!/bin/sh
### -- LSF job: Fine-tune SepFormer 3→7 sources with LoRA on Libri7Mix --
### -- Requires submit_dataprep7.sh to complete first. --
#BSUB -q gpua100
#BSUB -J sepformer7
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=40GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o analysis/dl_separation/logs/sepformer7_%J.out
#BSUB -e analysis/dl_separation/logs/sepformer7_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/dl_separation/logs

module load python3/3.12.11
module load cuda/12.6

VENV=.venv_gpu
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet --upgrade pip

if ! python3 -c "import torchaudio; torchaudio.load" 2>/dev/null; then
    echo "Installing PyTorch + torchaudio (cu126)..."
    pip install --quiet --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

if ! python3 -c "import speechbrain" 2>/dev/null; then
    pip install --quiet speechbrain
fi

if ! python3 -c "import soundfile" 2>/dev/null; then
    pip install --quiet soundfile
fi

# Verify training data exists
if [ ! -d "$BLACKHOLE/libri7mix/train-360/mix" ]; then
    echo "ERROR: $BLACKHOLE/libri7mix/train-360/mix not found."
    echo "Run submit_dataprep7.sh first and wait for it to complete."
    exit 1
fi

nvidia-smi

python3 analysis/dl_separation/hpc_finetune_sepformer7.py \
    --train-dir  $BLACKHOLE/libri7mix/train-360 \
    --val-dir    $BLACKHOLE/libri7mix/dev \
    --base-model speechbrain/sepformer-wsj03mix \
    --base-n-src 3 \
    --sr         16000 \
    --epochs     30 \
    --batch-size 8 \
    --num-workers 4 \
    --ckpt-dir   analysis/dl_separation/logs/sepformer7_ckpt
