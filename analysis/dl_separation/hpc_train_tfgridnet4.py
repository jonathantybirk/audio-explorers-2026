"""
HPC: Train TF-GridNet from scratch for 4-source separation on Libri4Mix.

Architecture: TF-GridNet (Luo & Mesgarani, ICASSP 2023)
  - STFT frontend (n_fft=256, hop=128, 8kHz)
  - Embedding conv: 2 (re+im) → D channels
  - N GridNet blocks, each with:
      IntraFrame BiLSTM  — processes F frequency bins per time step
      InterFrame BiLSTM  — processes T time steps per frequency bin
  - Output conv: D → n_src * 2 (real+imaginary masks per source)
  - Complex masking + ISTFT

Why TF-GridNet over SepFormer:
  - Operates on complex STFT (explicit phase modelling)
  - Grid structure natively handles T×F jointly
  - 23.1 dB SI-SNRi on WSJ0-2Mix vs SepFormer's 22.4 dB (best published single model)
  - Training from scratch on 4-source data sidesteps the surgery problem entirely

Dataset: Libri4Mix (train-360, 8kHz, mix_clean)
         + on-the-fly RIR augmentation to match reverberant hearing-aid conditions.

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

Usage:
    pip install pyroomacoustics torch
    python hpc_train_tfgridnet4.py \\
        --train-dir /path/to/libri4mix/train-360 \\
        --val-dir   /path/to/libri4mix/dev \\
        --epochs 100 --batch-size 16

Approximate training time: ~24–48h on 1× H100 for 100 epochs.
For faster iteration use --d-model 48 --n-blocks 4 (smaller model, ~8h).
"""

# ── SLURM header ──────────────────────────────────────────────────────────────
# #SBATCH --job-name=tfgridnet4
# #SBATCH --gres=gpu:h100:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --time=48:00:00
# #SBATCH --output=tfgridnet4_%j.log

import argparse
import glob
import os
import warnings

import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ── TF-GridNet architecture ───────────────────────────────────────────────────

class IntraFrameModule(nn.Module):
    """BiLSTM over the frequency axis for each time frame independently.

    Input:  (B, D, T, F)
    Output: (B, D, T, F)  — residual
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, F = x.shape
        r = x.permute(0, 2, 3, 1).reshape(B * T, F, D)   # (B*T, F, D)
        r = self.norm(r)
        r, _ = self.lstm(r)
        r = self.proj(r)                                   # (B*T, F, D)
        r = r.reshape(B, T, F, D).permute(0, 3, 1, 2)    # (B, D, T, F)
        return x + r


class InterFrameModule(nn.Module):
    """BiLSTM over the time axis for each frequency bin independently.

    Input:  (B, D, T, F)
    Output: (B, D, T, F)  — residual
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, F = x.shape
        r = x.permute(0, 3, 2, 1).reshape(B * F, T, D)   # (B*F, T, D)
        r = self.norm(r)
        r, _ = self.lstm(r)
        r = self.proj(r)                                   # (B*F, T, D)
        r = r.reshape(B, F, T, D).permute(0, 3, 2, 1)    # (B, D, T, F)
        return x + r


class TFGridNetBlock(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.intra = IntraFrameModule(d_model, hidden)
        self.inter = InterFrameModule(d_model, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inter(self.intra(x))


class TFGridNet(nn.Module):
    """TF-GridNet for n_src-speaker separation.

    Args:
        n_src:    number of output sources
        n_fft:    STFT window size
        hop:      STFT hop size
        d_model:  embedding dimension
        hidden:   BiLSTM hidden size (per direction)
        n_blocks: number of TFGridNetBlocks
    """

    def __init__(
        self,
        n_src: int = 4,
        n_fft: int = 256,
        hop: int = 128,
        d_model: int = 64,
        hidden: int = 192,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.n_src  = n_src
        self.n_fft  = n_fft
        self.hop    = hop
        self.n_bins = n_fft // 2 + 1

        # 2 input channels (real, imaginary)
        self.input_proj  = nn.Conv2d(2, d_model, kernel_size=1)
        self.blocks      = nn.ModuleList([TFGridNetBlock(d_model, hidden) for _ in range(n_blocks)])
        # Output: n_src complex masks → n_src * 2 real channels (re + im per source)
        self.output_proj = nn.Conv2d(d_model, n_src * 2, kernel_size=1)
        self.d_model = d_model

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """Separate mix into n_src sources.

        Args:
            mix: (B, T_samples) mono waveform, float32

        Returns:
            sources: (B, n_src, T_samples)
        """
        window = torch.hann_window(self.n_fft, device=mix.device)
        # X: (B, F, T_frames) complex
        X = torch.stft(mix, self.n_fft, self.hop, window=window,
                       return_complex=True, center=False)

        # Stack real + imaginary → (B, 2, F, T_frames)
        inp = torch.stack([X.real, X.imag], dim=1)

        # (B, 2, F, T) → (B, D, F, T)
        feat = self.input_proj(inp)
        # Rearrange to (B, D, T, F) for LSTM processing
        feat = feat.permute(0, 1, 3, 2)

        for block in self.blocks:
            feat = block(feat)

        # Back to (B, D, F, T)
        feat = feat.permute(0, 1, 3, 2)
        # (B, n_src*2, F, T)
        masks = self.output_proj(feat)
        # (B, n_src, 2, F, T)
        masks = masks.view(mix.shape[0], self.n_src, 2, self.n_bins, -1)

        sources = []
        for s in range(self.n_src):
            m_re = masks[:, s, 0]   # (B, F, T)
            m_im = masks[:, s, 1]
            # Complex multiplication: (X_re + j*X_im) * (m_re + j*m_im)
            s_re = X.real * m_re - X.imag * m_im
            s_im = X.real * m_im + X.imag * m_re
            S = torch.complex(s_re, s_im)
            wav = torch.istft(S, self.n_fft, self.hop, window=window,
                              length=mix.shape[-1], center=False)
            sources.append(wav)

        return torch.stack(sources, dim=1)   # (B, n_src, T)


# ── PIT SI-SNR loss ───────────────────────────────────────────────────────────

def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR between estimate and target. Higher is better."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target   = target   - target.mean(dim=-1, keepdim=True)
    dot      = (estimate * target).sum(dim=-1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    e_noise  = estimate - s_target
    return 10 * torch.log10(
        s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps) + eps
    )


def pit_si_snr_loss(estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant SI-SNR loss (minimise negative mean SI-SNRi).

    Args:
        estimates: (B, n_src, T)
        targets:   (B, n_src, T)

    Returns:
        scalar loss (negative mean SI-SNR, lower is better separation)
    """
    import itertools
    B, n_src, T = targets.shape
    perms = list(itertools.permutations(range(n_src)))

    best_loss = None
    for perm in perms:
        perm_t = torch.stack([targets[:, p] for p in perm], dim=1)   # (B, n_src, T)
        scores = torch.stack(
            [si_snr(estimates[:, s], perm_t[:, s]) for s in range(n_src)], dim=1
        )   # (B, n_src)
        loss = -scores.mean()
        if best_loss is None or loss < best_loss:
            best_loss = loss

    return best_loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class Libri4MixDataset(Dataset):
    """Libri4Mix (mix_clean layout) with on-the-fly RIR augmentation."""

    def __init__(self, root: str, sr: int = 8000, max_len_s: float = 4.0, augment: bool = True):
        self.sr      = sr
        self.max_len = int(max_len_s * sr)
        self.augment = augment
        self.mix_files = sorted(glob.glob(os.path.join(root, "mix_clean", "*.wav")))
        assert len(self.mix_files) > 0, (
            f"No WAV files found in {root}/mix_clean/ — check data path."
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
        room_dim = rng.uniform([3.0, 3.0, 2.5], [8.0, 8.0, 4.0])
        mic_pos  = room_dim / 2 + rng.uniform(-0.3, 0.3, 3)
        src_pos  = room_dim / 2 + rng.uniform(-1.5, 1.5, 3)
        src_pos  = np.clip(src_pos, 0.15, room_dim - 0.15)
        rt60     = rng.uniform(0.15, 0.6)
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
        srcs = [self._load(os.path.join(os.path.dirname(os.path.dirname(self.mix_files[idx])),
                                        f"s{k}", stem)) for k in range(1, 5)]

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

        sources = torch.from_numpy(np.stack(rev_srcs, axis=0))   # (4, T)
        mix = sources.sum(dim=0)
        peak = mix.abs().max() + 1e-8
        return mix / peak, sources / peak


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TFGridNet(
        n_src=4,
        n_fft=args.n_fft,
        hop=args.hop,
        d_model=args.d_model,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

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
        model.train()
        train_loss = 0.0
        for mix, sources in train_dl:
            mix, sources = mix.to(device), sources.to(device)
            estimates = model(mix)                    # (B, 4, T)
            loss = pit_si_snr_loss(estimates, sources)
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
                estimates = model(mix)
                val_loss += pit_si_snr_loss(estimates, sources).item()

        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train SI-SNRi {-train_loss:.2f} dB  "
            f"val SI-SNRi {-val_loss:.2f} dB  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt)
            print(f"  → Saved checkpoint  val SI-SNRi {-best_val:.2f} dB")

    print(f"\nDone. Best val SI-SNRi: {-best_val:.2f} dB")
    print(f"Checkpoint: {os.path.abspath(args.ckpt_dir)}/best_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TF-GridNet 4-source separator from scratch")
    p.add_argument("--train-dir",   required=True)
    p.add_argument("--val-dir",     required=True)
    p.add_argument("--sr",          type=int,   default=8000)
    p.add_argument("--n-fft",       type=int,   default=256,
                   help="STFT window size (256 = 32ms at 8kHz)")
    p.add_argument("--hop",         type=int,   default=128,
                   help="STFT hop size (128 = 16ms at 8kHz)")
    p.add_argument("--d-model",     type=int,   default=64,
                   help="Embedding dimension. 64 = standard, 48 = faster (~8h on H100)")
    p.add_argument("--hidden",      type=int,   default=192,
                   help="BiLSTM hidden size per direction")
    p.add_argument("--n-blocks",    type=int,   default=6,
                   help="Number of TFGridNetBlocks. 6 = standard, 4 = faster")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max-len-s",   type=float, default=4.0)
    p.add_argument("--num-workers", type=int,   default=8)
    p.add_argument("--ckpt-dir",    default="./tfgridnet4_checkpoint")
    args = p.parse_args()
    train(args)
