# HPC Reference

## GPU compatibility

| GPU       | VRAM  | TF-GridNet batch | SepFormer+LoRA batch |
|-----------|-------|------------------|----------------------|
| H100      | 80 GB | 16               | 24                   |
| A100      | 80 GB | 16               | 24                   |
| A100      | 40 GB | 12               | 16                   |
| V100      | 32 GB | 8                | 12                   |
| V100      | 16 GB | 4                | 8                    |

All three GPUs work. V100 just needs smaller batches — adjust `--batch-size` accordingly.
Also update the `#SBATCH --gres=` line in each script to match what you get.

---

## One-time data prep (run once on cluster)

```bash
# Download LibriSpeech
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf train-clean-360.tar.gz
tar -xzf dev-clean.tar.gz

# Generate Libri4Mix at 8kHz
git clone https://github.com/JorisCos/LibriMix && cd LibriMix
pip install -r requirements.txt
python scripts/create_librimix_from_metadata.py \
    --librispeech_dir /path/to/LibriSpeech \
    --metadata_dir metadata/Libri4Mix \
    --n_src 4 \
    --out_dir /path/to/libri4mix \
    --freqs 8k \
    --modes min \
    --types mix_clean
```

---

## Option A — SepFormer + LoRA (~8–12h on A100/H100, ~24h on V100)

Best starting point: pretrained 3-source SepFormer fine-tuned to 4 sources.
Only 2–5% of parameters trained. Good if you want results quickly.

```bash
pip install speechbrain pyroomacoustics

# H100 / A100 80GB
python hpc_finetune_sepformer4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 30 \
    --batch-size 24

# A100 40GB
python hpc_finetune_sepformer4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 30 \
    --batch-size 16

# V100 32GB
python hpc_finetune_sepformer4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 30 \
    --batch-size 12

# V100 16GB
python hpc_finetune_sepformer4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 30 \
    --batch-size 8
```

Checkpoint saved to `./sepformer4_lora_checkpoint/best_model.pt`

---

## Option B — TF-GridNet from scratch (~24–48h on A100/H100, ~72h on V100)

Better architecture, trains from scratch. Architecturally superior to SepFormer.

```bash
pip install pyroomacoustics torch

# H100 / A100 80GB — full model
python hpc_train_tfgridnet4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 100 \
    --batch-size 16

# A100 40GB / V100 32GB — full model, smaller batch
python hpc_train_tfgridnet4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 100 \
    --batch-size 8

# V100 16GB — smaller model (~8h), use this if time is short
python hpc_train_tfgridnet4.py \
    --train-dir /path/to/libri4mix/train-360 \
    --val-dir   /path/to/libri4mix/dev \
    --epochs 100 \
    --batch-size 4 \
    --d-model 48 \
    --n-blocks 4
```

Checkpoint saved to `./tfgridnet4_checkpoint/best_model.pt`

---

## SLURM submission

Update the `--gres` line in each script before submitting:

```bash
# H100
#SBATCH --gres=gpu:h100:1

# A100
#SBATCH --gres=gpu:a100:1

# V100
#SBATCH --gres=gpu:v100:1

sbatch hpc_finetune_sepformer4.py   # Option A
sbatch hpc_train_tfgridnet4.py      # Option B
```
