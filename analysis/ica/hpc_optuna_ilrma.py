"""
Optuna hyperparameter optimisation for ILRMA on mixture.wav.

ILRMA (Independent Low-Rank Matrix Analysis) is a determined BSS algorithm:
it assumes n_src == n_mics (here both = 4).  It models each source's power
spectrogram as a low-rank NMF product and jointly optimises the demixing
matrix W together with the NMF bases via alternating IP updates.

Searches over:
  stft_size    : {512, 1024, 2048}
  hop_size     : stft_size // {2, 4}
  n_iter       : 20–150
  n_components : 2–16  (NMF rank for source spectra)
  proj_back    : True / False

Objective: minimise sum of |off-diagonal| cross-correlations across 4
separated sources (lower = more independent = better).

Outputs:
  analysis/ica/optuna/best_params_ilrma.json
  analysis/ica/separated/mixture_ilrma_opt_source_{1..4}_{az}deg.wav
"""

import argparse
import itertools
import json
import os
import warnings

import numpy as np
import optuna
import pyroomacoustics as pra
from scipy.io import wavfile

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "separated")
OPT_DIR   = os.path.join(os.path.dirname(__file__), "optuna")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser()
_parser.add_argument("--n_trials", type=int, default=80)
_parser.add_argument("--n_jobs",   type=int, default=1)
_args = _parser.parse_args()

# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [ D/2,  L/2], [ D/2, -L/2],
    [-D/2,  L/2], [-D/2, -L/2],
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))


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


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def srp_phat_doa(channels, sr, azimuths=np.linspace(0, 360, 360, endpoint=False)):
    n = min(len(ch) for ch in channels)
    channels = [ch[:n] for ch in channels]
    n_fft = 1 << (n - 1).bit_length()
    gcc_store = {(a, b): gcc_phat(channels[a], channels[b], n_fft)
                 for a, b in ALL_PAIRS}
    power = np.zeros(len(azimuths))
    for idx, az in enumerate(azimuths):
        phi = np.deg2rad(az)
        for a, b in ALL_PAIRS:
            tau = ((MIC_POS[a, 0] - MIC_POS[b, 0]) * np.sin(phi)
                 + (MIC_POS[a, 1] - MIC_POS[b, 1]) * np.cos(phi)) / C
            lag = tau * sr + n_fft // 2
            lo = int(np.clip(np.floor(lag), 0, n_fft - 1))
            hi = int(np.clip(lo + 1,        0, n_fft - 1))
            frac = lag - np.floor(lag)
            power[idx] += (1 - frac) * gcc_store[(a, b)][lo] + frac * gcc_store[(a, b)][hi]
    return float(azimuths[np.argmax(power)])


def ilrma_separate(data, stft_size, hop_size, n_iter, n_components, proj_back):
    """Run ILRMA; returns (n_samples, 4) mono sources."""
    analysis_win  = pra.hann(stft_size)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop_size)
    X = pra.transform.stft.analysis(data, stft_size, hop_size, win=analysis_win)
    Y = pra.bss.ilrma(X, n_iter=n_iter, n_components=n_components, proj_back=proj_back)
    n_samples = data.shape[0]
    sources = []
    for k in range(Y.shape[2]):
        mono = pra.transform.stft.synthesis(Y[:, :, k], stft_size, hop_size, win=synthesis_win)
        sources.append(mono[:n_samples].real.astype(np.float64))
    return np.stack(sources, axis=1)


def cross_corr_sum(mono):
    n = mono.shape[1]
    corr = np.corrcoef(mono.T)
    return float(np.sum(np.abs(corr[~np.eye(n, dtype=bool)])))


# ── Load audio ────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | {data.shape[0]/sr:.1f}s\n")


def objective(trial):
    stft_size    = trial.suggest_categorical("stft_size",    [512, 1024, 2048])
    divisor      = trial.suggest_categorical("hop_divisor",  [2, 4])
    hop_size     = stft_size // divisor
    n_iter       = trial.suggest_int("n_iter",       20, 150, step=10)
    n_components = trial.suggest_int("n_components", 2,  16,  step=2)
    proj_back    = trial.suggest_categorical("proj_back",    [True, False])

    try:
        mono = ilrma_separate(data, stft_size, hop_size, n_iter, n_components, proj_back)
        return cross_corr_sum(mono)
    except Exception as e:
        print(f"  trial failed: {e}")
        return 999.0


# ── Run Optuna ────────────────────────────────────────────────────────────────
print(f"Running Optuna ({_args.n_trials} trials) ...")
study = optuna.create_study(
    direction="minimize",
    study_name="ilrma_mixture",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=_args.n_trials, n_jobs=_args.n_jobs, show_progress_bar=False)

best = study.best_params
best["hop_size"] = best["stft_size"] // best.pop("hop_divisor")
best["best_cross_corr"] = study.best_value
best_path = os.path.join(OPT_DIR, "best_params_ilrma.json")
with open(best_path, "w") as f:
    json.dump(best, f, indent=2)

print(f"\nBest params (cross-corr {study.best_value:.4f}):")
for k, v in best.items():
    print(f"  {k}: {v}")
print(f"Saved → {os.path.relpath(best_path)}")

# ── Final separation with best params ─────────────────────────────────────────
print("\nRunning final separation with best params ...")
mono = ilrma_separate(data, best["stft_size"], best["hop_size"],
                      best["n_iter"], best["n_components"], best["proj_back"])

azimuths = np.linspace(0, 360, 360, endpoint=False)
for k in range(mono.shape[1]):
    az = srp_phat_doa([data[:, m] for m in range(4)], sr, azimuths)
    fname = f"mixture_ilrma_opt_source_{k+1}_{az:.0f}deg.wav"
    save_wav(os.path.join(OUT_DIR, fname), mono[:, k], sr)

print("\nDone.")
