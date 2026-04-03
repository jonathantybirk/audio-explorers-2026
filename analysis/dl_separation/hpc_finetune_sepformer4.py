"""
HPC job: fine-tune SepFormer-WHAMR from 2 → 4 speakers using WSJ0-4Mix.

Why this is the right HPC job
──────────────────────────────
No pretrained 4-speaker separation model exists in any mainstream framework.
The closest we have is SepFormer-WHAMR (2 speakers, reverberant training).
WSJ0-4Mix is the standard benchmark dataset for 4-speaker separation.

Fine-tuning rather than training from scratch:
  - SepFormer-WHAMR already knows reverberant speech separation.
  - We just need to extend its output head from 2 → 4 sources.
  - Expected to converge faster and generalise better than from scratch.

Running this script
────────────────────
This is written for SLURM-managed HPC clusters with A100/H100 GPUs.
Estimated training time: ~12–24h on 1× A100 (WSJ0-4Mix full set).

On HPC:
  # 1. Copy this file to the cluster
  # 2. pip install speechbrain asteroid
  # 3. Prepare WSJ0-4Mix (see below)
  # 4. Submit: sbatch hpc_finetune_sepformer4.py  (or run directly)

WSJ0-4Mix preparation (one-time on HPC)
────────────────────────────────────────
  pip install git+https://github.com/speechbrain/speechbrain.git
  # Download WSJ0 corpus (LDC93S6A, LDC94S13A) — requires LDC licence.
  # Then generate mixtures:
  #   python create_wsj0_mix.py --n-speakers 4 --wsj0-dir /path/to/wsj0 --output-dir /path/to/wsj04mix

  # Alternatively: Libri4Mix (no licence needed):
  pip install git+https://github.com/JorisCos/LibriMix
  # python create_librimix_from_metadata.py --librispeech-dir ... --n-src 4

Output
──────
Fine-tuned model checkpoint saved to ./sepformer4_checkpoint/
After training, copy back to mac and load with:
  model = SepformerSeparation.from_hparams(source="./sepformer4_checkpoint")

Usage
─────
  python hpc_finetune_sepformer4.py \\
      --train-dir /path/to/wsj04mix/tr \\
      --val-dir   /path/to/wsj04mix/cv \\
      --epochs    50 \\
      --batch-size 4
"""

# ── SLURM header (kept as comments so the script also runs directly) ──────────
# #SBATCH --job-name=sepformer4
# #SBATCH --gres=gpu:a100:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --time=24:00:00
# #SBATCH --output=sepformer4_%j.log

import argparse
import glob
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# SpeechBrain imports (no torchaudio stub needed on HPC — CUDA build is fine)
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper


# ── Dataset ───────────────────────────────────────────────────────────────────
class WSJ04MixDataset(Dataset):
    """
    Expects WSJ0-4Mix directory layout:
      mix/  — mixture WAVs
      s1/   — source 1 WAVs
      s2/   — source 2 WAVs
      s3/   — source 3 WAVs
      s4/   — source 4 WAVs
    """
    def __init__(self, root, sr=8000, max_len_s=4.0):
        self.root    = root
        self.sr      = sr
        self.max_len = int(max_len_s * sr)
        self.mix_files = sorted(glob.glob(os.path.join(root, "mix", "*.wav")))
        assert len(self.mix_files) > 0, f"No files in {root}/mix/"

    def _load(self, path):
        sr, sig = wavfile.read(path)
        if sig.dtype == np.int16:
            sig = sig.astype(np.float32) / 32768.0
        if len(sig) > self.max_len:
            sig = sig[:self.max_len]
        else:
            sig = np.pad(sig, (0, self.max_len - len(sig)))
        return torch.from_numpy(sig)

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        stem = os.path.basename(self.mix_files[idx])
        mix  = self._load(self.mix_files[idx])
        srcs = []
        for k in range(1, 5):
            src_path = os.path.join(self.root, f"s{k}", stem)
            srcs.append(self._load(src_path))
        sources = torch.stack(srcs, dim=0)   # (4, T)
        return mix, sources


# ── Model surgery: 2-source SepFormer → 4-source ─────────────────────────────
def extend_to_4_sources(model):
    """
    SepFormer's final mask-net output is (batch, 2, T).
    We replace the last linear layer to produce 4 masks instead of 2.
    All other weights are kept (leverages reverberant pretraining).
    """
    masknet = model.mods.masknet

    # Find the final convolutional / linear output layer
    # In SpeechBrain SepFormer this lives at masknet.output_layer or similar
    # We do a generic search for the layer that produces n_src outputs.
    for name, module in masknet.named_modules():
        if isinstance(module, nn.Conv1d) and module.out_channels == 2:
            print(f"  Extending output layer '{name}': 2 → 4 channels")
            old_w = module.weight.data   # (2, C_in, K)
            new_w = torch.cat([old_w, old_w], dim=0)   # (4, C_in, K) — duplicate
            module.out_channels = 4
            module.weight = nn.Parameter(new_w)
            if module.bias is not None:
                old_b = module.bias.data
                module.bias = nn.Parameter(torch.cat([old_b, old_b]))
            print(f"  Done. New output shape: {module.weight.shape}")
            return model

    raise RuntimeError("Could not find 2-channel output Conv1d in SepFormer masknet. "
                       "Check SpeechBrain version or model architecture.")


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained 2-source SepFormer-WHAMR
    print("Loading SepFormer-WHAMR pretrained weights ...")
    model_sb = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir="./sepformer_whamr_pretrained",
        run_opts={"device": str(device)},
    )

    # Extend output from 2 → 4 sources
    model_sb = extend_to_4_sources(model_sb)
    model    = model_sb.mods.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    train_ds  = WSJ04MixDataset(args.train_dir, sr=8000, max_len_s=args.max_len_s)
    val_ds    = WSJ04MixDataset(args.val_dir,   sr=8000, max_len_s=args.max_len_s)
    train_dl  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
    val_dl    = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    best_val  = float("inf")
    ckpt_dir  = "./sepformer4_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for mix, sources in train_dl:
            mix, sources = mix.to(device), sources.to(device)

            # Forward: (B, T) → (B, 4, T)
            est = model_sb.mods.encoder(mix.unsqueeze(1))
            est = model_sb.mods.masknet(est)
            est = model_sb.mods.decoder(est)
            if est.dim() == 2:
                est = est.unsqueeze(1)

            loss = get_si_snr_with_pitwrapper(sources, est)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, sources in val_dl:
                mix, sources = mix.to(device), sources.to(device)
                est = model_sb.mods.encoder(mix.unsqueeze(1))
                est = model_sb.mods.masknet(est)
                est = model_sb.mods.decoder(est)
                if est.dim() == 2:
                    est = est.unsqueeze(1)
                val_loss += get_si_snr_with_pitwrapper(sources, est).item()

        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train SI-SNRi {-train_loss:.2f} dB  val SI-SNRi {-val_loss:.2f} dB")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  → Saved best checkpoint (val SI-SNRi {-best_val:.2f} dB)")

    print(f"\nTraining complete. Best val SI-SNRi: {-best_val:.2f} dB")
    print(f"Checkpoint: {os.path.abspath(ckpt_dir)}/best_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune SepFormer 2-src → 4-src on WSJ0-4Mix")
    p.add_argument("--train-dir",   required=True,  help="Path to WSJ0-4Mix training set")
    p.add_argument("--val-dir",     required=True,  help="Path to WSJ0-4Mix validation set")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--max-len-s",   type=float, default=4.0,
                   help="Max audio clip length in seconds (4s = standard for WSJ0-Mix)")
    args = p.parse_args()
    train(args)
