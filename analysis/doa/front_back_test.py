"""
Front/back disambiguation for near-axis sources.

Uses sub-sample GCC-PHAT between each isolated source and the
raw LF vs LR channels. Parabolic interpolation around the peak
gives ~0.1-sample resolution — enough to resolve the ~0.77-sample
difference between 0° and 180° given the 6mm intra-ear baseline.

Sign convention:
  lag(source→LF) - lag(source→LR) > 0  →  LR closer  →  BACK  (180°)
  lag(source→LF) - lag(source→LR) < 0  →  LF closer  →  FRONT (0°)
"""
import os, json
import numpy as np
from scipy.io import wavfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_DIR   = os.path.join(REPO_ROOT, "analysis", "ica", "separated")
RAW_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH", "Software Case", "mixture.wav")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")

with open(GEO_PATH) as f:
    geo = json.load(f)
C = geo["speed_of_sound_m_s"]
L = geo["intra_ear_spacing_m"]   # 6mm front-rear baseline

EXPECTED_TDOA_SAMPLES = None  # computed after loading sr

def load_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:  d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2**31
    if d.ndim > 1: d = d.mean(axis=1)
    return sr, d.astype(np.float64)

def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= (np.abs(G) + 1e-12)
    cc = np.fft.fftshift(np.fft.irfft(G, n=n_fft)).real
    return cc

def sub_sample_peak(cc):
    """Parabolic interpolation around the integer peak."""
    n = len(cc)
    k = np.argmax(cc)
    if k == 0 or k == n - 1:
        return float(k - n // 2)
    # parabolic fit
    alpha = cc[k - 1]
    beta  = cc[k]
    gamma = cc[k + 1]
    delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-12)
    return float((k + delta) - n // 2)

def measure_front_back(src, lf, lr, sr):
    """Returns (lag_lf - lag_lr) in samples. Negative = front, positive = back."""
    n = min(len(src), len(lf), len(lr))
    n_fft = 1 << (n - 1).bit_length()

    lag_lf = sub_sample_peak(gcc_phat(src[:n], lf[:n], n_fft))
    lag_lr = sub_sample_peak(gcc_phat(src[:n], lr[:n], n_fft))
    return lag_lf - lag_lr

print("Loading mixture.wav ...")
sr, mix2 = wavfile.read(RAW_PATH)
if mix2.dtype == np.int16: mix2 = mix2.astype(np.float64) / 32768.0
elif mix2.dtype == np.int32: mix2 = mix2.astype(np.float64) / 2**31
lf = mix2[:, 0]
lr = mix2[:, 1]

expected = (L / C) * sr  # samples: ~0.77 at 44100 Hz
print(f"  Expected TDOA magnitude for 0° vs 180°: ±{expected:.3f} samples\n")

# Sources to test — (label, filename)
SOURCES = [
    ("Mountain man     (inter-ear n=3 s3)", "mixture_interear_n3_source_3.wav"),
    ("Convo man        (inter-ear n=4 s2)", "mixture_interear_n4_source_2.wav"),
    ("Convo man        (inter-ear n=4 s3)", "mixture_interear_n4_source_3.wav"),
    ("Ageing man       (inter-ear n=5 s1)", "mixture_interear_n5_source_1.wav"),
    ("Burning house    (bw bss n=4 s3)",    "brunch_woman_bss_n4_source_3.wav"),
    ("Australia man    (bss n=2 s2)",        "australia_man_bss_n2_source_2.wav"),
    ("Brunch woman     (bss n=4 s3)",        "brunch_woman_bss_n4_source_3.wav"),
]

print(f"{'Source':<45} {'lag_LF-LR':>10}  {'verdict':>8}")
print("-" * 70)
for label, fname in SOURCES:
    path = os.path.join(SEP_DIR, fname)
    if not os.path.exists(path):
        path2 = os.path.join(REPO_ROOT, "analysis", "doa", fname)
        if not os.path.exists(path2):
            print(f"{label:<45}  FILE NOT FOUND")
            continue
        path = path2
    _, src = load_wav(path)
    diff = measure_front_back(src, lf, lr, sr)
    verdict = "FRONT (0°)" if diff < 0 else "BACK (180°)"
    print(f"{label:<45} {diff:>+10.3f}  {verdict}")

print(f"\n  Threshold: expected ±{expected:.3f} samples for a clean source.")
print(f"  Values near 0 = ambiguous. |diff| >> {expected:.2f} = strong signal.")
