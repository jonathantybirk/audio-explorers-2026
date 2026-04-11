"""
Intra-ear GCC-PHAT: LF vs LR (left ear, 1.6 cm front-back baseline).

For a source at   0° (front): LF leads LR → GCC peak at positive lag
For a source at 180° (back):  LR leads LF → GCC peak at negative lag

Runs on each fmnmf2-opt-n5 mono source to get front/back sign per source.
Also runs on the raw mixture for reference.
"""

import os
import numpy as np
from scipy.io import wavfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    return sr, data.astype(np.float64)

def gcc_phat(x, y):
    n_fft = 1 << (len(x) + len(y) - 1).bit_length()
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= (np.abs(G) + 1e-12)
    cc = np.fft.irfft(G, n=n_fft)
    # fftshift so lag=0 is at centre
    return np.fft.fftshift(cc), n_fft

def peak_lag_samples(x, y, max_lag_samples=20):
    """Return the lag (samples) at which GCC-PHAT peaks, restricted to ±max_lag."""
    cc, n_fft = gcc_phat(x, y)
    centre = n_fft // 2
    lo = centre - max_lag_samples
    hi = centre + max_lag_samples + 1
    window = cc[lo:hi]
    peak_idx = np.argmax(window)
    lag = peak_idx - max_lag_samples          # positive = x leads y = front
    peak_val = window[peak_idx]
    return lag, peak_val, window

# ── Load raw mixture: channel order LF=0, LR=1, RF=2, RR=3 ──────────────────
sr, mix = load_wav(RAW_PATH)
LF_raw = mix[:, 0]
LR_raw = mix[:, 1]

print(f"Sample rate: {sr} Hz")
print(f"1.6 cm baseline → max TDOA = {0.016/343*sr:.2f} samples\n")

# ── Raw mixture reference ────────────────────────────────────────────────────
lag, val, _ = peak_lag_samples(LF_raw, LR_raw)
sign = "FRONT (LF leads)" if lag > 0 else ("BACK (LR leads)" if lag < 0 else "AMBIGUOUS (lag=0)")
print(f"Raw mixture   lag={lag:+d} samples  peak={val:.4f}  → {sign}")

print()
print("─" * 60)
print("fmnmf2-opt-n5 sources — intra-ear GCC (LF vs LR)")
print("─" * 60)

SOURCES = {
    1: "ageing man dominant",
    2: "australia man dominant",
    3: "mountain + burning house men",
    4: "brunch woman dominant",
    5: "convo couple",
}

for n in range(1, 6):
    path = os.path.join(SEP_DIR, f"mixture_fmnmf2_opt_n5_source_{n}.wav")
    sr2, mono = load_wav(path)
    assert sr2 == sr

    # Cross-correlate mono source with each raw intra-ear channel pair
    lag_lf_lr, val_lf_lr, win = peak_lag_samples(LF_raw, LR_raw)

    # Better: use the mono source as a reference filter via cross-corr with raw channels
    # GCC(mono_src, LF) vs GCC(mono_src, LR): whichever is higher tells us
    # which mic the source arrived at first
    n_fft = 1 << (len(mono) + len(LF_raw) - 1).bit_length()
    S  = np.fft.rfft(mono,   n=n_fft)
    LF = np.fft.rfft(LF_raw, n=n_fft)
    LR = np.fft.rfft(LR_raw, n=n_fft)

    def gcc_peak(A, B, max_lag=20):
        G = A * np.conj(B)
        G /= (np.abs(G) + 1e-12)
        cc = np.fft.fftshift(np.fft.irfft(G, n=n_fft))
        centre = n_fft // 2
        w = cc[centre - max_lag: centre + max_lag + 1]
        idx = np.argmax(w)
        return idx - max_lag, w[idx]

    lag_s_lf, amp_lf = gcc_peak(S, LF)
    lag_s_lr, amp_lr = gcc_peak(S, LR)

    # The mic with higher GCC amplitude = where source arrived first
    # Positive TDOA (LF leads LR) = source from front
    tdoa_samples = lag_s_lr - lag_s_lf   # positive if LF leads (front)
    amp_ratio    = amp_lf / (amp_lr + 1e-9)

    if   tdoa_samples > 0:  direction = "FRONT"
    elif tdoa_samples < 0:  direction = "BACK"
    else:                   direction = "AMBIGUOUS"

    print(f"Source {n} ({SOURCES[n]})")
    print(f"  lag(src→LF)={lag_s_lf:+d}  amp_LF={amp_lf:.4f}")
    print(f"  lag(src→LR)={lag_s_lr:+d}  amp_LR={amp_lr:.4f}")
    print(f"  TDOA LF-LR = {tdoa_samples:+d} samples  amp_ratio={amp_ratio:.3f}  → {direction}")
    print()
