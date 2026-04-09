"""
Optuna hyperparameter optimisation for FastMNMF2 on mixture.wav.

Searches over:
  stft_size   : {1024, 2048, 4096}
  hop_size    : stft_size // {2, 4, 8}
  n_iter      : 30–150
  n_components: 4–16
  n_src       : 5, 6, 7, 8  (or fixed via --n_src)

Objective: minimise sum of absolute off-diagonal cross-correlations across
separated sources (lower = more independent = better separation).

Saves best params per n_src to:
  analysis/ica/optuna/best_params_nsrc{N}.json
and runs a final separation with best params, writing outputs to
  analysis/ica/separated/mixture_fmnmf2_opt_n{N}_source_*.wav
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
_parser.add_argument("--n_src",     type=int,   default=None,
                     help="Fix n_src (default: search 5–8)")
_parser.add_argument("--n_trials",  type=int,   default=50,
                     help="Optuna trials per n_src value")
_parser.add_argument("--n_jobs",    type=int,   default=1,
                     help="Parallel Optuna workers (set to n CPU cores)")
_args = _parser.parse_args()

# ── Geometry ──────────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [ D/2,  L/2],
    [ D/2, -L/2],
    [-D/2,  L/2],
    [-D/2, -L/2],
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


def fastmnmf2_separate(data, stft_size, hop_size, n_iter, n_components, n_src):
    analysis_win  = pra.hann(stft_size)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop_size)
    X = pra.transform.stft.analysis(data, stft_size, hop_size, win=analysis_win)
    Y_all = pra.bss.fastmnmf2(X, n_src=n_src, n_iter=n_iter,
                               n_components=n_components, mic_index="all")
    n_sources = Y_all.shape[3]
    n_samples  = data.shape[0]

    mono_sources = []
    for k in range(n_sources):
        Y_k_avg = Y_all[:, :, :, k].mean(axis=0)
        mono = pra.transform.stft.synthesis(Y_k_avg, stft_size, hop_size, win=synthesis_win)
        mono_sources.append(mono[:n_samples].real.astype(np.float64))

    return np.stack(mono_sources, axis=1)   # (n_samples, n_sources)


def cross_corr_sum(mono_sources):
    """Sum of |off-diagonal| cross-correlations — lower = better."""
    n = mono_sources.shape[1]
    corr = np.corrcoef(mono_sources.T)
    mask = ~np.eye(n, dtype=bool)
    return float(np.sum(np.abs(corr[mask])))


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa(ch_a, ch_b, phi_rad):
    return ((MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]) * np.sin(phi_rad)
          + (MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]) * np.cos(phi_rad)) / C


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
            tau = predicted_tdoa(a, b, phi)
            lag = tau * sr + n_fft // 2
            lo = int(np.clip(np.floor(lag), 0, n_fft - 1))
            hi = int(np.clip(lo + 1,        0, n_fft - 1))
            frac = lag - np.floor(lag)
            power[idx] += (1 - frac) * gcc_store[(a, b)][lo] + frac * gcc_store[(a, b)][hi]
    return float(azimuths[np.argmax(power)])


# ── Load audio once ───────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | {data.shape[0]/sr:.1f}s\n")

NSRC_LIST = [_args.n_src] if _args.n_src else [5, 6, 7, 8]


def make_objective(n_src_fixed):
    def objective(trial):
        stft_size   = trial.suggest_categorical("stft_size",    [1024, 2048, 4096])
        divisor     = trial.suggest_categorical("hop_divisor",  [2, 4, 8])
        hop_size    = stft_size // divisor
        n_iter      = trial.suggest_int("n_iter",       30, 150, step=10)
        n_components= trial.suggest_int("n_components", 4,  16,  step=2)

        try:
            mono = fastmnmf2_separate(data, stft_size, hop_size, n_iter,
                                      n_components, n_src_fixed)
            score = cross_corr_sum(mono)
        except Exception as e:
            print(f"  trial failed: {e}")
            return 999.0

        return score
    return objective


# ── Optimise per n_src ────────────────────────────────────────────────────────
for n_src in NSRC_LIST:
    best_path = os.path.join(OPT_DIR, f"best_params_nsrc{n_src}.json")
    print(f"\n{'='*70}")
    print(f"  Optimising n_src={n_src}  ({_args.n_trials} trials, {_args.n_jobs} workers)")
    print(f"{'='*70}")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"fastmnmf2_nsrc{n_src}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(n_src), n_trials=_args.n_trials,
                   n_jobs=_args.n_jobs, show_progress_bar=False)

    best = study.best_params
    best["n_src"] = n_src
    best["hop_size"] = best["stft_size"] // best.pop("hop_divisor")
    best["best_cross_corr"] = study.best_value

    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Best params (cross-corr {study.best_value:.4f}):")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"  Saved → {os.path.relpath(best_path)}")

    # ── Final separation with best params ─────────────────────────────────────
    print(f"\n  Running final separation with best params ...")
    mono = fastmnmf2_separate(
        data,
        stft_size    = best["stft_size"],
        hop_size     = best["hop_size"],
        n_iter       = best["n_iter"],
        n_components = best["n_components"],
        n_src        = n_src,
    )

    azimuths = np.linspace(0, 360, 360, endpoint=False)

    # Re-synthesise per-mic images for proper DoA estimation
    analysis_win  = pra.hann(best["stft_size"])
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, best["hop_size"])
    X = pra.transform.stft.analysis(data, best["stft_size"], best["hop_size"], win=analysis_win)
    Y_all = pra.bss.fastmnmf2(X, n_src=n_src, n_iter=best["n_iter"],
                               n_components=best["n_components"], mic_index="all")
    n_samples = data.shape[0]

    for k in range(Y_all.shape[3]):
        chans = []
        for mic in range(Y_all.shape[0]):
            sig = pra.transform.stft.synthesis(
                Y_all[mic, :, :, k], best["stft_size"], best["hop_size"], win=synthesis_win)
            chans.append(sig[:n_samples].real.astype(np.float64))
        az = srp_phat_doa(chans, sr, azimuths)
        fname = f"mixture_fmnmf2_opt_n{n_src}_source_{k+1}_{az:.0f}deg.wav"
        save_wav(os.path.join(OUT_DIR, fname), mono[:, k], sr)

print("\nDone.")
