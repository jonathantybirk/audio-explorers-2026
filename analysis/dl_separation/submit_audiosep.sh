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

# AudioSep has no setup.py/pyproject.toml — clone and add to PYTHONPATH
AUDIOSEP_DIR="$HOME/AudioSep"
if [ ! -d "$AUDIOSEP_DIR" ]; then
    echo "Cloning AudioSep..."
    git clone --depth 1 https://github.com/Audio-AGI/AudioSep "$AUDIOSEP_DIR" || {
        echo "ERROR: git clone of AudioSep failed." >&2; exit 1
    }
fi
# Install AudioSep's dependencies
pip install --quiet transformers huggingface_hub peft librosa soundfile laion-clap

# Install anything in the repo's own requirements.txt
if [ -f "$AUDIOSEP_DIR/requirements.txt" ]; then
    pip install --quiet -r "$AUDIOSEP_DIR/requirements.txt"
fi

export PYTHONPATH="$AUDIOSEP_DIR:$PYTHONPATH"

# AudioSep class lives in pipeline.py, NOT an audiosep package
if ! python3 -c "from pipeline import AudioSep" 2>/dev/null; then
    echo "ERROR: AudioSep import still fails after clone. Check $AUDIOSEP_DIR." >&2
    echo "Contents of $AUDIOSEP_DIR:" >&2
    ls "$AUDIOSEP_DIR" >&2
    echo "Python path: $PYTHONPATH" >&2
    python3 -c "from pipeline import AudioSep" >&2  # print the actual error
    exit 1
fi
echo "AudioSep import OK"

nvidia-smi

echo "=== AudioSep inference on mixture.wav ==="
python3 analysis/dl_separation/hpc_audiosep_inference.py \
    --n_speakers 7 \
    --device cuda

echo "=== Done ==="
