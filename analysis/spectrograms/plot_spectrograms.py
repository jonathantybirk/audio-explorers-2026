"""
Plot spectrograms for all 4 channels of both mixture WAV files.
Images saved to data/spectrograms/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

CHANNEL_NAMES = ["Left Front", "Left Rear", "Right Front", "Right Rear"]
WAV_FILES = {
    "mixture": "DONT-TOUCH/Software Case/mixture.wav",
    "example_mixture": "DONT-TOUCH/Software Case/example_mixture.wav",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_DIR = os.path.join(REPO_ROOT, "data", "spectrograms")
os.makedirs(SAVE_DIR, exist_ok=True)

NPERSEG = 1024
NOVERLAP = 768
FMAX_HZ = 8000  # speech lives below 8 kHz


def load_wav(rel_path):
    sr, data = wavfile.read(os.path.join(REPO_ROOT, rel_path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    else:
        data = data.astype(np.float32)
    return sr, data


def plot_spectrograms(name, sr, data):
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle(name, fontsize=13, y=1.01)

    for ch, ax in enumerate(axes):
        f, t, Sxx = spectrogram(data[:, ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)

        # Trim to FMAX_HZ
        f_mask = f <= FMAX_HZ
        f, Sxx = f[f_mask], Sxx[f_mask]

        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        vmax = np.percentile(Sxx_db, 99)
        vmin = vmax - 60  # 60 dB dynamic range

        ax.imshow(
            Sxx_db,
            origin="lower",
            aspect="auto",
            extent=[t[0], t[-1], f[0] / 1000, f[-1] / 1000],
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("kHz", fontsize=9)
        ax.text(0.01, 0.92, CHANNEL_NAMES[ch], transform=ax.transAxes,
                fontsize=9, color="white", va="top")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()

    out = os.path.join(SAVE_DIR, f"{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


for name, rel_path in WAV_FILES.items():
    sr, data = load_wav(rel_path)
    print(f"{name}: sr={sr} Hz, shape={data.shape}")
    plot_spectrograms(name, sr, data)
