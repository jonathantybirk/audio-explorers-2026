"""
Optuna hyperparameter search for FastMNMF2 n=4 on the inter-ear masked mixture.
Objective: minimise pairwise cross-correlation sum (lower = better separation).

Self-contained: loads mixture.wav directly and applies inter-ear balance mask inline.
"""
import os, warnings
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import stft, istft
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
os.makedirs(SEP_DIR, exist_ok=True)

N_TRIALS  = 60
N_SRC     = 4
NPERSEG   = 2048
NOVERLAP  = NPERSEG - 512
THRESH    = 0.15

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:   d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    return sr, d.astype(np.float64)

def save_wav(path, sig, sr):
    sig  = np.nan_to_num(sig)
    peak = np.max(np.abs(sig)) + 1e-9
    out  = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}", flush=True)

# ── Load and apply inter-ear balance mask ─────────────────────────────────────
print(f"Loading {os.path.relpath(RAW_PATH)} ...", flush=True)
sr, mix = load_wav(RAW_PATH)
N = mix.shape[0]
print(f"  {N} samples | {mix.shape[1]} ch | {sr} Hz | {N/sr:.1f}s", flush=True)

print("Applying inter-ear balance mask ...", flush=True)
Xm = np.stack([stft(mix[:, ch], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)[2] for ch in range(4)])
lf_pow     = np.abs(Xm[0])**2 + np.abs(Xm[1])**2
rf_pow     = np.abs(Xm[2])**2 + np.abs(Xm[3])**2
left_ratio = lf_pow / (lf_pow + rf_pow + 1e-9)
mask       = ((left_ratio >= (0.5 - THRESH)) & (left_ratio <= (0.5 + THRESH))).astype(np.float64)
print(f"  Kept {mask.mean()*100:.1f}% of TF bins", flush=True)

data = np.zeros((N, 4), dtype=np.float64)
for ch in range(4):
    _, sig = istft(Xm[ch] * mask, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    data[:, ch] = sig[:N].real
print(f"  Masked 4-channel signal ready\n", flush=True)

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    stft_size = trial.suggest_categorical("stft_size", [512, 1024, 2048, 4096])
    hop_size  = trial.suggest_categorical("hop_size",  [128, 256, 512, 1024])
    n_iter    = trial.suggest_int("n_iter",  50, 300, step=50)
    n_comp    = trial.suggest_categorical("n_comp", [4, 6, 8])

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
        cc_sum = float(np.sum(np.abs(np.corrcoef(srcs.T)[~np.eye(N_SRC, dtype=bool)])))
        print(f"  trial {trial.number}: stft={stft_size} hop={hop_size} iter={n_iter} comp={n_comp} → cc={cc_sum:.4f}", flush=True)
        return cc_sum
    except Exception as e:
        print(f"  trial {trial.number} failed: {e}", flush=True)
        return 999.0

study = optuna.create_study(
    direction="minimize",
    study_name=f"fastmnmf2_interear_n{N_SRC}",
    sampler=optuna.samplers.TPESampler(seed=42),
)
print(f"Running {N_TRIALS} Optuna trials for FastMNMF2 n={N_SRC} ...", flush=True)
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

best = study.best_params
print(f"\nBest params: {best}", flush=True)
print(f"Best cross-corr sum: {study.best_value:.4f}", flush=True)

# ── Re-run best params and save ───────────────────────────────────────────────
print("\nRe-running best params and saving sources ...", flush=True)
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
    print(f"  Source {k+1}: RMS={rms:.4f}", flush=True)
    save_wav(os.path.join(SEP_DIR, f"mixture_interear_opt_n4_source_{k+1}.wav"), mono, sr)

print("\nDone.", flush=True)
