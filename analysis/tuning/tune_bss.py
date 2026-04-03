"""
Bayesian hyperparameter tuning for AuxIVA and ILRMA using Optuna.

Blind objective — no ground-truth DoA is assumed:
  1. Cross-correlation score: sum of absolute off-diagonal entries in the
     pairwise correlation matrix of separated mono sources. Lower = better
     separation (sources more statistically independent).
  2. DNSMOS score: mean OVRL score across separated sources, estimated by
     Microsoft DNSMOS (blind, no reference signal). Higher = better perceptual
     speech quality.

Combined objective (maximised):
    score = -w_corr * corr_score + w_mos * mean_mos

Both weights are configurable at the top of this file.

Usage:
    python analysis/tuning/tune_bss.py --method auxiva --n-trials 80
    python analysis/tuning/tune_bss.py --method ilrma  --n-trials 80
    python analysis/tuning/tune_bss.py --method both   --n-trials 80

Results are saved to analysis/tuning/results/{method}_study.csv and
the best hyperparameters printed at the end.
"""

import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import resample_poly

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV_PATH = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "example_mixture.wav")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Objective weights ─────────────────────────────────────────────────────────
W_CORR = 1.0   # weight on cross-correlation penalty (higher = punish correlation more)
W_MOS  = 0.5   # weight on DNSMOS OVRL score

DNSMOS_SR = 16000  # DNSMOS requires 16 kHz input


# ── Audio loading ─────────────────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


# ── Separation functions ──────────────────────────────────────────────────────
def run_auxiva(data, sr, n_fft, hop, n_iter):
    analysis_win  = pra.hann(n_fft)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop)
    X = pra.transform.stft.analysis(data, n_fft, hop, win=analysis_win)
    Y, _ = pra.bss.auxiva(
        X, n_src=data.shape[1], n_iter=n_iter, proj_back=False, return_filters=True,
    )
    gains = pra.bss.projection_back(Y, X.mean(axis=2))
    Y_mono = Y * gains[None, :, :]
    sources = pra.transform.stft.synthesis(Y_mono, n_fft, hop, win=synthesis_win)
    sources = sources[: data.shape[0], :].real.astype(np.float64)
    return sources  # (n_samples, n_sources)


def run_ilrma(data, sr, n_fft, hop, n_iter, n_components):
    analysis_win  = pra.hann(n_fft)
    synthesis_win = pra.transform.stft.compute_synthesis_window(analysis_win, hop)
    X = pra.transform.stft.analysis(data, n_fft, hop, win=analysis_win)
    Y, _ = pra.bss.ilrma(
        X,
        n_src=data.shape[1],
        n_iter=n_iter,
        proj_back=False,
        n_components=n_components,
        return_filters=True,
    )
    gains = pra.bss.projection_back(Y, X.mean(axis=2))
    Y_mono = Y * gains[None, :, :]
    sources = pra.transform.stft.synthesis(Y_mono, n_fft, hop, win=synthesis_win)
    sources = sources[: data.shape[0], :].real.astype(np.float64)
    return sources  # (n_samples, n_sources)


# ── Scoring ───────────────────────────────────────────────────────────────────
def cross_corr_score(sources):
    """Sum of absolute off-diagonal entries of the pairwise correlation matrix."""
    corr = np.corrcoef(sources.T)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.sum(np.abs(corr[mask])))


def dnsmos_score(sources, sr):
    """Mean OVRL DNSMOS score across all separated sources (blind, no reference)."""
    from speechmos import dnsmos

    # Resample to 16 kHz if needed
    if sr != DNSMOS_SR:
        gcd = np.gcd(sr, DNSMOS_SR)
        up, down = DNSMOS_SR // gcd, sr // gcd
        resampled = [resample_poly(sources[:, k], up, down) for k in range(sources.shape[1])]
    else:
        resampled = [sources[:, k] for k in range(sources.shape[1])]

    scores = []
    for sig in resampled:
        # Normalise to [-1, 1] to avoid clipping artefacts in MOS estimate
        peak = np.max(np.abs(sig)) + 1e-12
        sig_norm = sig / peak * 0.9
        result = dnsmos.run(sig_norm.astype(np.float32), DNSMOS_SR, return_df=False)
        scores.append(result["ovrl_mos"])

    return float(np.mean(scores))


def objective_score(sources, sr):
    """Combined objective (higher is better)."""
    corr = cross_corr_score(sources)
    mos  = dnsmos_score(sources, sr)
    return -W_CORR * corr + W_MOS * mos


# ── Optuna objectives ─────────────────────────────────────────────────────────
def make_auxiva_objective(data, sr):
    def objective(trial):
        n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048, 4096])
        hop   = trial.suggest_categorical("hop_fraction", [4, 2])  # n_fft // hop_fraction
        n_iter = trial.suggest_int("n_iter", 20, 200, step=10)

        hop_size = n_fft // hop

        try:
            sources = run_auxiva(data, sr, n_fft, hop_size, n_iter)
        except Exception as e:
            # Raise TrialPruned so Optuna skips silently rather than crashing
            raise optuna.TrialPruned(f"auxiva failed: {e}")

        return objective_score(sources, sr)

    return objective


def make_ilrma_objective(data, sr):
    def objective(trial):
        n_fft       = trial.suggest_categorical("n_fft", [512, 1024, 2048, 4096])
        hop         = trial.suggest_categorical("hop_fraction", [4, 2])
        n_iter      = trial.suggest_int("n_iter", 20, 200, step=10)
        n_components = trial.suggest_int("n_components", 2, 8)

        hop_size = n_fft // hop

        try:
            sources = run_ilrma(data, sr, n_fft, hop_size, n_iter, n_components)
        except Exception as e:
            raise optuna.TrialPruned(f"ilrma failed: {e}")

        return objective_score(sources, sr)

    return objective


# ── Study runner ──────────────────────────────────────────────────────────────
def run_study(method, data, sr, n_trials):
    print(f"\n{'='*60}")
    print(f"  Tuning {method.upper()}  |  {n_trials} trials")
    print(f"  Objective: -{W_CORR}*corr_score + {W_MOS}*dnsmos_ovrl")
    print(f"{'='*60}")

    if method == "auxiva":
        objective = make_auxiva_objective(data, sr)
    else:
        objective = make_ilrma_objective(data, sr)

    study = optuna.create_study(direction="maximize",
                                study_name=f"{method}_tuning",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save all trial results
    df = study.trials_dataframe()
    csv_path = os.path.join(OUT_DIR, f"{method}_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  All trials saved → {os.path.relpath(csv_path)}")

    best = study.best_trial
    print(f"\n  Best trial #{best.number}  score={best.value:.4f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        if k == "hop_fraction":
            n_fft = best.params.get("n_fft", "?")
            print(f"    {k}: {v}  (hop_size = {n_fft}//{v} = {n_fft//v if isinstance(n_fft, int) else '?'})")
        else:
            print(f"    {k}: {v}")

    return study


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSS hyperparameter tuning via Optuna")
    parser.add_argument("--method", choices=["auxiva", "ilrma", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=80)
    args = parser.parse_args()

    print(f"Loading {os.path.relpath(WAV_PATH)} ...")
    sr, data = load_wav(WAV_PATH)
    print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | {data.shape[0]/sr:.1f}s")

    methods = ["auxiva", "ilrma"] if args.method == "both" else [args.method]

    studies = {}
    for method in methods:
        studies[method] = run_study(method, data, sr, args.n_trials)

    print("\n" + "="*60)
    print("  SUMMARY — best hyperparameters")
    print("="*60)
    for method, study in studies.items():
        best = study.best_trial
        params = best.params.copy()
        if "hop_fraction" in params:
            params["hop_size"] = params["n_fft"] // params.pop("hop_fraction")
        print(f"\n  {method.upper()}")
        for k, v in params.items():
            print(f"    {k}: {v}")
        print(f"    objective score: {best.value:.4f}")
