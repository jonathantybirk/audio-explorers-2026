#!/bin/sh
### -- LSF job: AudioSep zero-shot speaker separation on mixture.wav --
### -- Requires GPU. Downloads ~1GB AudioSep checkpoint on first run. --
#BSUB -q gpua100
#BSUB -J audiosep
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/audiosep_%J.out
#BSUB -e /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs/audiosep_%J.err

mkdir -p /zhome/53/3/169791/audio-explorers-2026/analysis/dl_separation/logs

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

if [ ! -f "DONT-TOUCH/Software Case/mixture.wav" ]; then
    echo "ERROR: mixture.wav not found." >&2; exit 1
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
    pip install --quiet --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

pip install --quiet soundfile
if ! python3 -c "from audiosep import AudioSep" 2>/dev/null; then
    echo "Installing AudioSep from GitHub..."
    pip install --quiet git+https://github.com/Audio-AGI/AudioSep
fi

nvidia-smi

echo "=== AudioSep inference on mixture.wav ==="
python3 analysis/dl_separation/hpc_audiosep_inference.py \
    --n_speakers 7 \
    --device cuda

echo "=== Done ==="
