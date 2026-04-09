#!/bin/sh
### -- LSF job script for SepFormer 4-src LoRA fine-tune on DTU HPC --
#BSUB -q gpua100
#BSUB -J sepformer4
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/sepformer4_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/sepformer4_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

if [ -z "$BLACKHOLE" ] || [ ! -d "$BLACKHOLE" ]; then
    echo "ERROR: \$BLACKHOLE not set or not mounted on this node." >&2; exit 1
fi
if [ ! -d "$BLACKHOLE/libri4mix/train-360" ] || [ ! -d "$BLACKHOLE/libri4mix/dev" ]; then
    echo "ERROR: libri4mix data not found in \$BLACKHOLE. Run submit_dataprep.sh first." >&2; exit 1
fi

module load python3/3.12.11
module load cuda/12.6

if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed — no GPU available on this node." >&2; exit 1
fi

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

if ! python3 -c "import pyroomacoustics" 2>/dev/null; then
    pip install --quiet pyroomacoustics
fi

nvidia-smi

# Fine-tune speechbrain/sepformer-wsj03mix (19.8 dB SI-SNRi, 3-src) → 4 sources via LoRA
# Stronger base than hahmadraz (8.88 dB). Data is 8kHz, matches this model.
python3 analysis/dl_separation/hpc_finetune_sepformer4.py \
    --train-dir $BLACKHOLE/libri4mix/train-360 \
    --val-dir   $BLACKHOLE/libri4mix/dev \
    --base-model speechbrain/sepformer-wsj03mix \
    --base-n-src 3 \
    --sr 8000 \
    --epochs 30 \
    --batch-size 24 \
    --num-workers 4 \
    --ckpt-dir analysis/dl_separation/logs/sepformer4_ckpt
