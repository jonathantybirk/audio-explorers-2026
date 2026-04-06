# DTU HPC — Setup & Submission Notes

## 1. Connect via VSCode (recommended)

1. Connect to DTU VPN — must spoof the AnyConnect user-agent or the server rejects with "Please upgrade your AnyConnect Client":
   ```bash
   sudo openconnect --protocol=anyconnect \
     --useragent="AnyConnect Linux_64 4.10.07073" \
     vpn.dtu.dk
   ```
2. In VSCode, click the **blue "><" button** in the bottom-left corner.
3. Select **Connect to Host** → **Add New SSH Host**.
4. Enter: `s216136@login1.gbar.dtu.dk`
5. Platform: **Linux** — enter your DTU password when prompted.

---

## 2. Clone the repo on the cluster

The project name has a space locally. Use a clean name on the cluster:

```bash
cd ~
git clone git@github.com:YOUR_ORG/audio-explorers-2026.git audio-explorers-2026
```

Set up GitHub SSH keys first if you haven't:
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

---

## 3. One-time: prepare Libri4Mix data

Store on `/work3` — not `/zhome` — to avoid quota issues (zhome is 30 GB).

```bash
mkdir -p /work3/s216136/libri4mix

cd /work3/s216136
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf train-clean-360.tar.gz
tar -xzf dev-clean.tar.gz

git clone https://github.com/JorisCos/LibriMix
cd LibriMix
pip install -r requirements.txt
python scripts/create_librimix_from_metadata.py \
    --librispeech_dir /work3/s216136/LibriSpeech \
    --metadata_dir metadata/Libri4Mix \
    --n_src 4 \
    --out_dir /work3/s216136/libri4mix \
    --freqs 8k \
    --modes min \
    --types mix_clean
```

---

## 4. Queue & resource choices

| Queue | GPU | VRAM |
|---|---|---|
| **`gpua100`** | A100 | 40 / 80 GB |
| `gpuv100` | V100 | 16 / 32 GB — use if A100 queue is long |
| `gpul40s` | L40s | 48 GB |

Switch queue by changing `#BSUB -q gpua100` → `#BSUB -q gpuv100` in the submit scripts.

---

## 5. Submit a job

```bash
cd ~/audio-explorers-2026

# TF-GridNet training (~24–48 h)
bsub < analysis/dl_separation/submit_tfgridnet.sh

# SepFormer LoRA fine-tune (~16–20 h)
bsub < analysis/dl_separation/submit_sepformer.sh
```

### Monitor

```bash
bstat
bjobs <JOBID>
tail -f analysis/dl_separation/logs/tfgridnet4_<JOBID>.out
bkill <JOBID>
```

| State | Meaning |
|---|---|
| `PEND` | Waiting for a free GPU node |
| `RUN` | Running — output streams to `.out` file |
| `DONE` | Finished OK |
| `EXIT` | Error — check `.err` file |

---

## 6. Module & CUDA versions

```bash
module load python3/3.12.11
module load cuda/12.6
```

PyTorch wheel: `--index-url https://download.pytorch.org/whl/cu126`. If you change the CUDA module, change the wheel suffix to match.

---

## 7. Disk management

Venv and pip cache are on `/work3` to avoid filling zhome:
- Venv: `/work3/s216136/venv`
- Pip cache: `/work3/s216136/.pip-cache`
- Pretrained models: `/work3/s216136/pretrained_models`

```bash
getquota_zhome.sh              # check /zhome quota (30 GB)
rm -rvf ~/.cache/pip/http/*    # clear pip cache if needed
```

---

## 8. Common issues

### "Please upgrade your AnyConnect Client"
Use the spoofed user-agent command in section 1 — plain `sudo openconnect vpn.dtu.dk` won't work.

### Job stays PEND a long time
Switch to `gpuv100` — more nodes, shorter queue.

### Disk quota exceeded during pip install
The venv and pip cache are on `/work3` to prevent this. If it still happens, check `/work3` quota with `du -sh /work3/s216136/`.

### `libcudart.so` version mismatch
CUDA module must match the PyTorch wheel suffix (`cu126` ↔ `cuda/12.6`).
