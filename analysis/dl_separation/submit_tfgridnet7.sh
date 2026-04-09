#!/bin/sh
### -- LSF job: Train TF-GridNet 7-source from scratch on Libri7Mix --
### -- Requires submit_dataprep7.sh to have completed first. --
#BSUB -q gpua100
#BSUB -J tfgridnet7
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=40GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/tfgridnet7_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/tfgridnet7_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

if [ -z "$BLACKHOLE" ] || [ ! -d "$BLACKHOLE" ]; then
    echo "ERROR: \$BLACKHOLE not set or not mounted on this node." >&2; exit 1
fi
if [ ! -d "$BLACKHOLE/libri7mix/train-360/mix" ]; then
    echo "ERROR: libri7mix not found. Run submit_dataprep7.sh first." >&2; exit 1
fi

module load python3/3.12.11
module load cuda/12.6

if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed." >&2; exit 1
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

pip install --quiet soundfile scipy

nvidia-smi

echo "=== TF-GridNet 7-source training ==="
python3 analysis/dl_separation/hpc_train_tfgridnet7.py \
    --train-dir  $BLACKHOLE/libri7mix/train-360 \
    --val-dir    $BLACKHOLE/libri7mix/dev \
    --sr         16000 \
    --epochs     50 \
    --batch-size 4 \
    --num-workers 4 \
    --ckpt-dir   analysis/dl_separation/logs/tfgridnet7_ckpt

echo "=== Done ==="
