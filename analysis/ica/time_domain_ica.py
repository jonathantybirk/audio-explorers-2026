"""
Time-domain ICA baseline on example_mixture.wav.

This is the simple instantaneous-mixing baseline:

    x(t) = A s(t)

It uses sklearn FastICA directly on the 4 microphone channels without an STFT.
That model is intentionally simpler than AuxIVA and does not try to recover
spatially faithful source images or reliable DoA labels. The component order is
therefore arbitrary; files are saved as numbered components only.

Outputs saved to analysis/ica/separated/:
  tdica_component_1.wav
  tdica_component_2.wav
  tdica_component_3.wav
  tdica_component_4.wav
  tdica_spectrograms.png
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav")
OUT_DIR = os.path.join(os.path.dirname(__file__), "separated")
os.makedirs(OUT_DIR, exist_ok=True)

N_COMPONENTS = 4
MAX_ITER = 2000
TOL = 1e-5
RANDOM_STATE = 42


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def save_wav(path, signal, sr):
    peak = np.max(np.abs(signal)) + 1e-12
    out = np.clip(signal / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")


print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(
    f"  {data.shape[0]} samples  |  {data.shape[1]} channels  "
    f"|  {sr} Hz  |  {data.shape[0] / sr:.1f} s\n"
)

print(
    "Running FastICA "
    f"(components={N_COMPONENTS}, max_iter={MAX_ITER}, tol={TOL}) ..."
)
ica = FastICA(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE,
    max_iter=MAX_ITER,
    tol=TOL,
    whiten="unit-variance",
)
sources = ica.fit_transform(data)

if ica.n_iter_ < MAX_ITER:
    print(f"  Converged in {ica.n_iter_} iterations.")
else:
    print(f"  WARNING: FastICA hit the iteration limit ({ica.n_iter_}).")

print("\nSeparation diagnostics:")
print("  Source RMS amplitudes:")
for k in range(sources.shape[1]):
    rms = np.sqrt(np.mean(sources[:, k] ** 2))
    print(f"    Component {k}: RMS = {rms:.4f}")

print("  Pairwise cross-correlation (off-diagonal near 0 is good):")
corr = np.corrcoef(sources.T)
for i in range(corr.shape[0]):
    row = "    " + "  ".join(f"{corr[i, j]:+.3f}" for j in range(corr.shape[1]))
    print(row)

print("\nSaving separated audio ...")
for stale_path in glob.glob(os.path.join(OUT_DIR, "tdica_component_*.wav")):
    os.remove(stale_path)

for k in range(sources.shape[1]):
    path = os.path.join(OUT_DIR, f"tdica_component_{k + 1}.wav")
    save_wav(path, sources[:, k], sr)

print("\nPlotting spectrograms ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for k in range(sources.shape[1]):
    ax = axes[k // 2][k % 2]
    ax.specgram(sources[:, k], Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    ax.set_title(f"TD-ICA component {k + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
plt.suptitle("Time-domain FastICA components — example_mixture.wav", fontsize=13)
plt.tight_layout()
spec_path = os.path.join(OUT_DIR, "tdica_spectrograms.png")
plt.savefig(spec_path, dpi=150)
plt.close()
print(f"  saved  {os.path.relpath(spec_path)}")

print("\n════════════════════════════════════════════════════════════════════════")
print("  Time-domain ICA baseline — example_mixture.wav")
print(f"  FastICA converged in {ica.n_iter_} iterations")
print("  Notes:")
print("    - instantaneous-mixing baseline only")
print("    - component order is arbitrary")
print("    - no reliable DoA labels are assigned here")
print("════════════════════════════════════════════════════════════════════════")
