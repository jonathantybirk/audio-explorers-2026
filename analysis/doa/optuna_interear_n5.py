"""
Optuna hyperparameter search for FastMNMF2 n=5 on the inter-ear masked mixture.
Objective: minimise pairwise cross-correlation sum (lower = better separation).
"""
import os, warnings
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
IN_PATH   = os.path.join(SEP_DIR, "mixture_interear_masked.wav")
N_TRIALS  = 60
N_SRC     = 5

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    return sr, d.astype(np.float64)

def save_wav(path, sig, sr):
    sig  = np.nan_to_num(sig)
    peak = np.max(np.abs(sig)) + 1e-9
    out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")

print("Loading inter-ear masked mixture ...")
sr, data = load_wav(IN_PATH)
N = data.shape[0]
print(f"  {N} samples  {sr} Hz  {N/sr:.1f}s\n")

def objective(trial):
    stft_size  = trial.suggest_categorical("stft_size",  [512, 1024, 2048, 4096])
    hop_size   = trial.suggest_categorical("hop_size",   [128, 256, 512, 1024])
    n_iter     = trial.suggest_int("n_iter",  50, 300, step=50)
    n_comp     = trial.suggest_categorical("n_comp", [4, 6, 8])

    if hop_size >= stft_size:
        return 999.0

    try:
        aw = pra.hann(stft_size)
        sw = pra.transform.stft.compute_synthesis_window(aw, hop_size)
        X  = pra.transform.stft.analysis(data, stft_size, hop_size, win=aw)
        Y  = pra.bss.fastmnmf2(X, n_src=N_SRC, n_iter=n_iter,
                                n_components=n_comp, mic_index="all")
        srcs = np.stack([
            pra.transform.stft.synthesis(Y[:,:,:,k].mean(axis=0),
                                          stft_size, hop_size, win=sw)[:N].real
            for k in range(N_SRC)
        ], axis=1)
        corr    = np.corrcoef(srcs.T)
        cc_sum  = float(np.sum(np.abs(corr[~np.eye(N_SRC, dtype=bool)])))
        return cc_sum
    except Exception as e:
        return 999.0

study = optuna.create_study(direction="minimize")
print(f"Running {N_TRIALS} Optuna trials for FastMNMF2 n={N_SRC} ...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
print(f"\nBest params: {best}")
print(f"Best cross-corr sum: {study.best_value:.4f}")

# ── Re-run best params and save ───────────────────────────────────────────────
print("\nRe-running best params and saving sources ...")
aw = pra.hann(best["stft_size"])
sw = pra.transform.stft.compute_synthesis_window(aw, best["hop_size"])
X  = pra.transform.stft.analysis(data, best["stft_size"], best["hop_size"], win=aw)
Y  = pra.bss.fastmnmf2(X, n_src=N_SRC, n_iter=best["n_iter"],
                        n_components=best["n_comp"], mic_index="all")

for k in range(N_SRC):
    mono = pra.transform.stft.synthesis(
        Y[:,:,:,k].mean(axis=0), best["stft_size"], best["hop_size"], win=sw
    )[:N].real
    rms = np.sqrt(np.mean(mono**2))
    print(f"  Source {k+1}: RMS={rms:.4f}")
    save_wav(os.path.join(SEP_DIR, f"mixture_interear_opt_n5_source_{k+1}.wav"), mono, sr)

print("\nDone.")
