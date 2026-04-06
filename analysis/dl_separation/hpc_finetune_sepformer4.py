"""
HPC: Fine-tune SepFormer (3-source → 4-source) with LoRA on Libri4Mix.

Base model: speechbrain/sepformer-wsj03mix  (SI-SNRi 19.8 dB on WSJ0-3Mix)
Why this base: best publicly available speech separation checkpoint.
              3→4 surgery is a smaller step than 2→4 (WHAMR base).

Alternative base (no surgery needed, already 4-source but weaker start):
  --base-model hahmadraz/sepformer-libri4mix  (SI-SNRi 8.88 dB, 48kHz model)
  If you use this, also pass --base-n-src 4 and --sr 48000.

LoRA:
  Only the transformer attention Linear layers are adapted (default r=8).
  Encoder, decoder, and frozen backbone weights are untouched.
  The extended output head (3→4) is fully trainable.
  This means ~2–5% of total parameters are trained → fast convergence.

Dataset: Libri4Mix (train-360, 8kHz, mix_clean)
         + on-the-fly RIR augmentation (RT60 0.15–0.6 s) to match the
           reverberant hearing-aid recording conditions.

Data prep (one-time on cluster):
    git clone https://github.com/JorisCos/LibriMix && cd LibriMix
    pip install -r requirements.txt
    python scripts/create_librimix_from_metadata.py \\
        --librispeech_dir /path/to/LibriSpeech \\
        --metadata_dir metadata/Libri4Mix \\
        --n_src 4 \\
        --out_dir /path/to/libri4mix \\
        --freqs 8k \\
        --modes min \\
        --types mix_clean
    # Produces: train-360/  dev/  test/  each with s1/ s2/ s3/ s4/ mix_clean/

Usage:
    pip install speechbrain pyroomacoustics
    python hpc_finetune_sepformer4.py \\
        --train-dir /path/to/libri4mix/train-360 \\
        --val-dir   /path/to/libri4mix/dev \\
        --epochs 30 --batch-size 24

Output:
    Checkpoint saved to --ckpt-dir (default ./sepformer4_lora_checkpoint/)
    Load on Mac:
        from speechbrain.inference.separation import SepformerSeparation
        model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj03mix", ...)
        # then load state_dict from best_model.pt and run separate_batch()
"""

# ── SLURM header ──────────────────────────────────────────────────────────────
# #SBATCH --job-name=sepformer4_lora
# #SBATCH --gres=gpu:h100:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --time=16:00:00
# #SBATCH --output=sepformer4_lora_%j.log

import argparse
import glob
import os
import warnings

import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn as nn
from scipy.io import wavfile
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation.

    The original weight W is frozen.  Only the low-rank factors A (in×r)
    and B (r×out) are trained.  Output = W(x) + scale * x @ A @ B.
    """
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(linear.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, linear.out_features))
        self.scale = alpha / r

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def inject_lora(module: nn.Module, r: int = 8, alpha: float = 16.0) -> int:
    """Recursively replace all nn.Linear layers with LoRA-wrapped versions.

    Returns the number of layers replaced.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            n += 1
        else:
            n += inject_lora(child, r=r, alpha=alpha)
    return n


def freeze_non_lora(model: nn.Module):
    """Freeze everything; LoRA params stay trainable because they were created fresh."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)


# ── Model surgery ─────────────────────────────────────────────────────────────

def extend_output_to_n_src(model: nn.Module, n_src_new: int) -> nn.Module:
    """Extend SepFormer's output head from n_src_old → n_src_new.

    Searches masknet for the final Conv1d or Linear whose output dimension
    equals n_src_old and duplicates heads to reach n_src_new.
    Prints the full masknet structure on first run so you can verify on the
    cluster log.
    """
    masknet = model.mods.masknet

    print("\n── SepFormer masknet output layers ──")
    for name, module in masknet.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            out_dim = module.out_channels if isinstance(module, nn.Conv1d) else module.out_features
            print(f"  {name:60s}  {type(module).__name__}  out={out_dim}")
    print("─────────────────────────────────────\n")

    def _try_extend(layer, n_new):
        if isinstance(layer, nn.Conv1d):
            n_old = layer.out_channels
            if n_old >= n_new:
                return False
            repeats = (n_new + n_old - 1) // n_old
            new_w = torch.cat([layer.weight.data] * repeats, dim=0)[:n_new]
            layer.out_channels = n_new
            layer.weight = nn.Parameter(new_w)
            if layer.bias is not None:
                layer.bias = nn.Parameter(
                    torch.cat([layer.bias.data] * repeats)[:n_new]
                )
            return True
        if isinstance(layer, nn.Linear):
            n_old = layer.out_features
            if n_old >= n_new:
                return False
            repeats = (n_new + n_old - 1) // n_old
            new_w = torch.cat([layer.weight.data] * repeats, dim=0)[:n_new]
            layer.out_features = n_new
            layer.weight = nn.Parameter(new_w)
            if layer.bias is not None:
                layer.bias = nn.Parameter(
                    torch.cat([layer.bias.data] * repeats)[:n_new]
                )
            return True
        return False

    # Prefer masknet.output_layer
    if hasattr(masknet, "output_layer"):
        if _try_extend(masknet.output_layer, n_src_new):
            print(f"  Extended masknet.output_layer → {n_src_new} sources")
            return model

    # Fallback: find last Conv1d/Linear with small out dim (the source head)
    candidate_name, candidate_mod = None, None
    for name, mod in masknet.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            out = mod.out_channels if isinstance(mod, nn.Conv1d) else mod.out_features
            if out < 16:  # output head has n_src outputs, not large feature dims
                candidate_name, candidate_mod = name, mod

    if candidate_mod is not None and _try_extend(candidate_mod, n_src_new):
        print(f"  Extended '{candidate_name}' → {n_src_new} sources")
        return model

    raise RuntimeError(
        f"Could not find output head to extend to {n_src_new} sources.\n"
        "Check the printed masknet structure above and extend manually."
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class Libri4MixDataset(Dataset):
    """Libri4Mix (mix_clean layout) with on-the-fly RIR augmentation.

    Directory layout expected:
        root/
          mix_clean/   ← 4-speaker mixture
          s1/          ← clean source 1
          s2/          ← clean source 2
          s3/          ← clean source 3
          s4/          ← clean source 4
    """

    def __init__(self, root: str, sr: int = 8000, max_len_s: float = 4.0, augment: bool = True):
        self.root = root
        self.sr = sr
        self.max_len = int(max_len_s * sr)
        self.augment = augment
        self.mix_files = sorted(glob.glob(os.path.join(root, "mix_clean", "*.wav")))
        assert len(self.mix_files) > 0, (
            f"No WAV files found in {root}/mix_clean/ — check your data path."
        )

    def _load(self, path: str) -> np.ndarray:
        _, sig = wavfile.read(path)
        if sig.dtype == np.int16:
            sig = sig.astype(np.float32) / 32768.0
        elif sig.dtype == np.int32:
            sig = sig.astype(np.float32) / 2**31
        else:
            sig = sig.astype(np.float32)
        if len(sig) > self.max_len:
            sig = sig[:self.max_len]
        else:
            sig = np.pad(sig, (0, self.max_len - len(sig)))
        return sig

    @staticmethod
    def _random_rir(sr: int, rng: np.random.Generator) -> np.ndarray:
        """Simulate a small-room RIR with random dimensions and RT60."""
        room_dim = rng.uniform([3.0, 3.0, 2.5], [8.0, 8.0, 4.0])
        mic_pos = room_dim / 2 + rng.uniform(-0.3, 0.3, 3)
        src_pos = room_dim / 2 + rng.uniform(-1.5, 1.5, 3)
        src_pos = np.clip(src_pos, 0.15, room_dim - 0.15)
        rt60 = rng.uniform(0.15, 0.6)
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
            room = pra.ShoeBox(
                room_dim, fs=sr,
                materials=pra.Material(e_abs),
                max_order=min(max_order, 12),
            )
            room.add_source(src_pos)
            room.add_microphone(mic_pos)
            room.simulate()
            rir = room.rir[0][0].astype(np.float32)
            return rir / (np.max(np.abs(rir)) + 1e-8)
        except Exception:
            return np.array([1.0], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.mix_files)

    def __getitem__(self, idx: int):
        stem = os.path.basename(self.mix_files[idx])
        srcs = [self._load(os.path.join(self.root, f"s{k}", stem)) for k in range(1, 5)]

        if self.augment:
            rng = np.random.default_rng(seed=idx)
            rev_srcs = []
            for s in srcs:
                rir = self._random_rir(self.sr, rng)
                rev = fftconvolve(s, rir)[: self.max_len]
                if len(rev) < self.max_len:
                    rev = np.pad(rev, (0, self.max_len - len(rev)))
                rev_srcs.append(rev.astype(np.float32))
        else:
            rev_srcs = srcs

        sources = torch.from_numpy(np.stack(rev_srcs, axis=0))  # (4, T)
        mix = sources.sum(dim=0)

        peak = mix.abs().max() + 1e-8
        return mix / peak, sources / peak


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Base model:  {args.base_model}  (n_src={args.base_n_src})")
    print(f"LoRA:        r={args.lora_r}, alpha={args.lora_alpha}")

    print(f"\nLoading {args.base_model} ...")
    pretrained_dir = os.environ.get(
        "PRETRAINED_DIR",
        f"./pretrained_{args.base_model.replace('/', '_')}",
    )
    model_sb = SepformerSeparation.from_hparams(
        source=args.base_model,
        savedir=pretrained_dir,
        run_opts={"device": str(device)},
    )

    # Extend output head if necessary
    if args.base_n_src < 4:
        print(f"\nExtending output head {args.base_n_src} → 4 sources ...")
        model_sb = extend_output_to_n_src(model_sb, n_src_new=4)

    # Inject LoRA into masknet transformer layers
    n_lora = inject_lora(model_sb.mods.masknet, r=args.lora_r, alpha=args.lora_alpha)
    print(f"\nInjected LoRA into {n_lora} Linear layers in masknet")

    # Freeze everything; LoRA params + extended output head stay trainable
    freeze_non_lora(model_sb.mods)

    # Unfreeze extended output head explicitly
    if hasattr(model_sb.mods.masknet, "output_layer"):
        for p in model_sb.mods.masknet.output_layer.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model_sb.mods.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model_sb.mods.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")

    model_sb.mods.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model_sb.mods.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = Libri4MixDataset(args.train_dir, sr=args.sr, max_len_s=args.max_len_s, augment=True)
    val_ds   = Libri4MixDataset(args.val_dir,   sr=args.sr, max_len_s=args.max_len_s, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model_sb.mods.train()
        train_loss = 0.0
        for mix, sources in train_dl:
            mix, sources = mix.to(device), sources.to(device)  # (B,T), (B,4,T)

            enc = model_sb.mods.encoder(mix.unsqueeze(1))
            masks = model_sb.mods.masknet(enc)
            # masks shape: (B, T, N, n_src) or (n_src, B, T, N) depending on SB version
            # Attempt both shapes robustly
            if masks.dim() == 4 and masks.shape[-1] == 4:
                # (B, T, N, n_src) — last dim is sources
                est = torch.stack(
                    [model_sb.mods.decoder(enc * masks[..., s]).squeeze(1) for s in range(4)],
                    dim=1,
                )
            elif masks.dim() == 4 and masks.shape[0] == 4:
                # (n_src, B, T, N)
                est = torch.stack(
                    [model_sb.mods.decoder(enc * masks[s]).squeeze(1) for s in range(4)],
                    dim=1,
                )
            else:
                # Fallback: pass masks directly to decoder (old SB API)
                est = model_sb.mods.decoder(masks)
                if est.dim() == 2:
                    est = est.unsqueeze(1)

            # Trim to same length as sources
            T = sources.shape[-1]
            est = est[..., :T]
            if est.shape[-1] < T:
                est = torch.nn.functional.pad(est, (0, T - est.shape[-1]))

            loss = get_si_snr_with_pitwrapper(sources, est)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_sb.mods.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ── Validate ──────────────────────────────────────────────────────────
        model_sb.mods.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, sources in val_dl:
                mix, sources = mix.to(device), sources.to(device)
                enc = model_sb.mods.encoder(mix.unsqueeze(1))
                masks = model_sb.mods.masknet(enc)
                if masks.dim() == 4 and masks.shape[-1] == 4:
                    est = torch.stack(
                        [model_sb.mods.decoder(enc * masks[..., s]).squeeze(1) for s in range(4)],
                        dim=1,
                    )
                elif masks.dim() == 4 and masks.shape[0] == 4:
                    est = torch.stack(
                        [model_sb.mods.decoder(enc * masks[s]).squeeze(1) for s in range(4)],
                        dim=1,
                    )
                else:
                    est = model_sb.mods.decoder(masks)
                    if est.dim() == 2:
                        est = est.unsqueeze(1)
                T = sources.shape[-1]
                est = est[..., :T]
                if est.shape[-1] < T:
                    est = torch.nn.functional.pad(est, (0, T - est.shape[-1]))
                val_loss += get_si_snr_with_pitwrapper(sources, est).item()

        val_loss /= len(val_dl)
        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train SI-SNRi {-train_loss:.2f} dB  "
            f"val SI-SNRi {-val_loss:.2f} dB  "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save(model_sb.mods.state_dict(), ckpt)
            print(f"  → Saved checkpoint  val SI-SNRi {-best_val:.2f} dB")

    print(f"\nDone. Best val SI-SNRi: {-best_val:.2f} dB")
    print(f"Checkpoint: {os.path.abspath(args.ckpt_dir)}/best_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune SepFormer 3→4 sources with LoRA")
    p.add_argument("--train-dir",    required=True)
    p.add_argument("--val-dir",      required=True)
    p.add_argument("--base-model",   default="speechbrain/sepformer-wsj03mix",
                   help="HuggingFace model ID for the pretrained SepFormer base. "
                        "Use 'hahmadraz/sepformer-libri4mix' for the 4-source base "
                        "(weaker but no surgery needed — also set --base-n-src 4 --sr 48000).")
    p.add_argument("--base-n-src",   type=int, default=3,
                   help="Number of sources in the base model (3 for wsj03mix, 4 for libri4mix)")
    p.add_argument("--sr",           type=int, default=8000,
                   help="Sample rate — must match base model (8000 for wsj03mix, 48000 for libri4mix)")
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch-size",   type=int, default=24)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--lora-r",       type=int, default=8,
                   help="LoRA rank — higher = more parameters, more capacity")
    p.add_argument("--lora-alpha",   type=float, default=16.0,
                   help="LoRA alpha — scales the LoRA output (effective scale = alpha/r)")
    p.add_argument("--max-len-s",    type=float, default=4.0)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--ckpt-dir",     default="./sepformer4_lora_checkpoint")
    args = p.parse_args()
    train(args)
