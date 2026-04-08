#!/bin/sh
### -- LSF job script: download LibriSpeech + generate Libri4Mix into $BLACKHOLE --
### -- Runs on a CPU node, no GPU needed. Submit this BEFORE the training jobs. --
#BSUB -q hpc
#BSUB -J dataprep
#BSUB -n 4
#BSUB -W 6:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o analysis/dl_separation/logs/dataprep_%J.out
#BSUB -e analysis/dl_separation/logs/dataprep_%J.err

if [ -n "$LS_SUBCWD" ]; then
  cd "$LS_SUBCWD" || exit 1
fi

mkdir -p analysis/dl_separation/logs

DATA=$BLACKHOLE/libri4mix
LIBRI=$BLACKHOLE/LibriSpeech
mkdir -p "$DATA" "$LIBRI"

echo "=== Downloading LibriSpeech ==="
cd "$BLACKHOLE"

if [ ! -d "$LIBRI/train-clean-360" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/train-clean-360.tar.gz
    tar -xzf train-clean-360.tar.gz
    rm train-clean-360.tar.gz
fi

if [ ! -d "$LIBRI/dev-clean" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xzf dev-clean.tar.gz
    rm dev-clean.tar.gz
fi

echo "=== Setting up LibriMix ==="
module load python3/3.12.11

VENV=$BLACKHOLE/.venv_dataprep
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet pandas scipy

if [ ! -d "$BLACKHOLE/LibriMix" ]; then
    git clone --quiet https://github.com/JorisCos/LibriMix "$BLACKHOLE/LibriMix"
fi
cd "$BLACKHOLE/LibriMix"
pip install --quiet -r requirements.txt

echo "=== Generating Libri4Mix ==="
python scripts/create_librimix_from_metadata.py \
    --librispeech_dir "$LIBRI" \
    --metadata_dir metadata/Libri4Mix \
    --n_src 4 \
    --out_dir "$DATA" \
    --freqs 8k \
    --modes min \
    --types mix_clean

echo "=== Done. Libri4Mix written to $DATA ==="
ls "$DATA"
