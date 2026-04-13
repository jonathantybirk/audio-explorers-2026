"""
Activity-weighted SRP-PHAT DoA estimation per speaker.

For each isolated source (mono BSS output), compute a frame-level
activity envelope. Use it to weight the SRP-PHAT map computed from
the full 4-channel mixture. Frames where the speaker is loudest
dominate the DoA estimate — avoiding the bias of cross-correlating
a BSS output directly with raw channels.

Convention: 0=front, 90=left, 180=back, 270=right (as in mic_geometry.json)
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
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = np.array([
    [ D/2,  L/2],   # LF
    [ D/2, -L/2],   # LR
    [-D/2,  L/2],   # RF
    [-D/2, -L/2],   # RR
], dtype=np.float64)

AZIMUTHS   = np.linspace(0, 360, 720, endpoint=False)
WIN_SIZE   = 2048
HOP        = 512
TOP_FRAC   = 0.15   # use top 15% most-active frames

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
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft)).real

def srp_phat_frame(channels, sr):
    """Compute SRP-PHAT map over all azimuths for one frame of 4-channel audio."""
    n = len(channels[0])
    n_fft = WIN_SIZE * 2
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    # precompute GCC-PHAT for all pairs
    gcc = {}
    for a, b in pairs:
        gcc[(a,b)] = gcc_phat(channels[a], channels[b], n_fft)
    center = n_fft // 2

    scores = np.zeros(len(AZIMUTHS))
    for idx, az in enumerate(AZIMUTHS):
        phi = np.deg2rad(az)
        s = 0.0
        for a, b in pairs:
            dx = MIC_POS[a,0] - MIC_POS[b,0]
            dy = MIC_POS[a,1] - MIC_POS[b,1]
            tau = (dx * np.sin(phi) + dy * np.cos(phi)) / C
            lag = int(round(tau * sr))
            lag = np.clip(lag + center, 0, n_fft - 1)
            s += gcc[(a,b)][lag]
        scores[idx] = s
    return scores

def estimate_doa(src_path, mix4ch, sr, label):
    _, src = load_wav(src_path)
    n = min(len(src), mix4ch.shape[0])
    src = src[:n]
    mix = mix4ch[:n]

    # Frame-level RMS of the source
    frames = range(0, n - WIN_SIZE, HOP)
    rms = np.array([np.sqrt(np.mean(src[i:i+WIN_SIZE]**2)) for i in frames])

    # Keep only top TOP_FRAC most-active frames
    thresh = np.quantile(rms, 1.0 - TOP_FRAC)
    active = np.where(rms >= thresh)[0]

    if len(active) == 0:
        print(f"  {label}: no active frames found")
        return None

    # Accumulate SRP-PHAT map over active frames
    acc = np.zeros(len(AZIMUTHS))
    for fi in active:
        i = fi * HOP
        chs = [mix[i:i+WIN_SIZE, ch] for ch in range(4)]
        acc += srp_phat_frame(chs, sr)

    best_az = (360 - AZIMUTHS[np.argmax(acc)]) % 360
    # Also report 2nd peak
    smoothed = acc.copy()
    peak1 = np.argmax(smoothed)
    smoothed[max(0,peak1-30):peak1+30] = -np.inf
    peak2_az = (360 - AZIMUTHS[np.argmax(smoothed)]) % 360

    return best_az, peak2_az, len(active)

# ── Load raw mixture (for lateral speakers) and masked mixture (for near-axis) ─
print("Loading mixture.wav ...")
sr_r, mix_raw = wavfile.read(RAW_PATH)
if mix_raw.dtype == np.int16:  mix_raw = mix_raw.astype(np.float64) / 32768.0
elif mix_raw.dtype == np.int32: mix_raw = mix_raw.astype(np.float64) / 2**31

print("Loading mixture_interear_masked.wav (AM+BW suppressed) ...")
masked_path = os.path.join(SEP_DIR, "mixture_interear_masked.wav")
sr_m, mix_masked = wavfile.read(masked_path)
if mix_masked.dtype == np.int16:  mix_masked = mix_masked.astype(np.float64) / 32768.0
elif mix_masked.dtype == np.int32: mix_masked = mix_masked.astype(np.float64) / 2**31

# Lateral speakers: use raw mixture (they're dominant, no interference issue)
# Near-axis speakers: use masked mixture (removes AM+BW so SRP sees the central sources)
SOURCES = [
    ("Australia man   ",  os.path.join(SEP_DIR, "australia_man_bss_n2_source_2.wav"),    mix_raw),
    ("Brunch woman    ",  os.path.join(SEP_DIR, "brunch_woman_interear.wav"),             mix_raw),
    ("Mountain man    ",  os.path.join(SEP_DIR, "mixture_interear_n3_source_3.wav"),      mix_masked),
    ("Convo man (s2)  ",  os.path.join(SEP_DIR, "mixture_interear_n4_source_2.wav"),      mix_masked),
    ("Convo man (s3)  ",  os.path.join(SEP_DIR, "mixture_interear_n4_source_3.wav"),      mix_masked),
    ("Ageing man      ",  os.path.join(SEP_DIR, "mixture_interear_n5_source_1.wav"),      mix_masked),
    ("Burning house   ",  os.path.join(SEP_DIR, "brunch_woman_bss_n4_source_3.wav"),      mix_masked),
    # Convo Woman candidates
    ("ConvoW(couple)  ",  os.path.join(SEP_DIR, "best_convo_couple_nr.wav"),              mix_masked),
    ("ConvoW(n4 s1)   ",  os.path.join(SEP_DIR, "mixture_interear_n4_source_1.wav"),      mix_masked),
    ("ConvoW(n4 s4)   ",  os.path.join(SEP_DIR, "mixture_interear_n4_source_4.wav"),      mix_masked),
]

print(f"\n{'Speaker':<20} {'DoA #1':>8}  {'DoA #2':>8}  {'active frames':>14}")
print("-" * 60)
for label, path, mix4 in SOURCES:
    if not os.path.exists(path):
        print(f"  {label}: FILE NOT FOUND")
        continue
    result = estimate_doa(path, mix4, sr_r, label)
    if result:
        az1, az2, nf = result
        print(f"{label:<20} {az1:>7.1f}°  {az2:>7.1f}°  {nf:>14} frames")

print("\nDone.")
