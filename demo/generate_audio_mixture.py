"""
Generate all demo audio for mixture.wav.

Speaker positions are unknown for this recording — angles are estimated from
the signal itself using SRP-PHAT (see analysis/srp_phat/srp_phat_scan_mixture.py).
Those estimated angles are loaded from analysis/srp_phat/mixture_angles.json.

Beamforming methods (D&S, MVDR, mask-MVDR) use the estimated angles.
Separation methods (AuxIVA, ILRMA) are geometry-free.
AuxIVA is emitted in two variants:
  - default: the original demo settings
  - tuned: settings transferred from the finished Optuna study on
    example_mixture.wav
ILRMA is also emitted in two variants:
  - default: the original demo settings
  - tuned: settings transferred from the finished Optuna study on
    example_mixture.wav

All outputs go to demo/audio_mixture/.

Run from repo root:
    python demo/generate_audio_mixture.py
"""

import itertools
import json
import os
import glob

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import istft, stft

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WAV_PATH  = os.path.join(REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav")
GEO_PATH  = os.path.join(REPO_ROOT, "data", "mic_geometry.json")
ANGLES_PATH = os.path.join(REPO_ROOT, "analysis", "srp_phat", "mixture_angles.json")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "audio_mixture")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Beamforming hyperparameters ───────────────────────────────────────────────
NPERSEG    = 512
HOP        = 256
DIAG_LOAD  = 1e-4    # MVDR regularisation

# ── AuxIVA hyperparameters ────────────────────────────────────────────────────
AUXIVA_VARIANTS = [
    {
        "key": "default",
        "title": "AuxIVA default",
        "prefix": "auxiva",
        "stft": 2048,
        "hop": 1024,
        "iters": 30,
        "note": "default demo hp",
    },
    {
        "key": "tuned",
        "title": "AuxIVA tuned",
        "prefix": "auxiva_tuned",
        "stft": 2048,
        "hop": 512,
        "iters": 90,
        "note": "tuned on example_mixture.wav",
    },
]

# ── ILRMA hyperparameters ─────────────────────────────────────────────────────
ILRMA_VARIANTS = [
    {
        "key": "default",
        "title": "ILRMA default",
        "prefix": "ilrma",
        "stft": 2048,
        "hop": 1024,
        "iters": 100,
        "n_components": 4,
        "note": "default demo hp",
    },
    {
        "key": "tuned",
        "title": "ILRMA tuned",
        "prefix": "ilrma_tuned",
        "stft": 2048,
        "hop": 512,
        "iters": 110,
        "n_components": 2,
        "note": "tuned on example_mixture.wav",
    },
]

MIC_LABELS = ["LF", "LR", "RF", "RR"]


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


# ── Load estimated angles ─────────────────────────────────────────────────────
with open(ANGLES_PATH) as f:
    angles_data = json.load(f)
ESTIMATED_ANGLES = angles_data["estimated_angles_deg"]
print(f"Estimated speaker angles (SRP-PHAT): {ESTIMATED_ANGLES}")
print("Note: positions are estimates — no ground truth for mixture.wav\n")

# Build direction labels
def angle_label(az):
    """Short readable label for an estimated angle."""
    return f"{az:.0f}deg"


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    return sr, data.astype(np.float64)


def save_wav(path, signal, sr):
    peak = np.max(np.abs(signal)) + 1e-9
    out  = np.clip(signal / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved  {os.path.relpath(path)}")


def remove_stale_outputs(pattern):
    for path in glob.glob(os.path.join(OUT_DIR, pattern)):
        os.remove(path)
        print(f"  removed stale  {os.path.relpath(path)}")


# ── Time-domain beamformers ───────────────────────────────────────────────────
def delay_and_sum(channels, phi, sr):
    N   = len(channels[0])
    out = np.zeros(N, dtype=np.float64)
    for m in range(4):
        tau  = -(MIC_POS[m, 0]*np.sin(phi) + MIC_POS[m, 1]*np.cos(phi)) / C
        d    = tau * sr
        lo   = int(np.floor(d)); frac = d - lo
        idx  = np.arange(N) + lo
        v    = (idx >= 0) & (idx < N - 1)
        s    = np.zeros(N, dtype=np.float64)
        s[v] = (1-frac)*channels[m][idx[v]] + frac*channels[m][idx[v]+1]
        out += s
    return out / 4.0


def mvdr_blocking(channels, phi, sr):
    """MVDR with blocking matrix — geometry-based, no mask."""
    freqs_hz = np.fft.rfftfreq(NPERSEG, 1.0 / sr)
    X = np.stack([
        stft(ch, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)[2]
        for ch in channels
    ])                                   # (4, F, T)
    Xt = X.transpose(1, 0, 2)           # (F, 4, T)
    n_freqs, n_mics, n_frames = Xt.shape

    tau = -(MIC_POS[:, 0]*np.sin(phi) + MIC_POS[:, 1]*np.cos(phi)) / C
    a   = np.exp(1j * 2*np.pi * freqs_hz[:, None] * tau[None, :])  # (F, M)

    a_energy = np.einsum("fi,fi->f", a.conj(), a).real
    B = (np.eye(n_mics)[None, :, :]
         - np.einsum("fi,fj->fij", a, a.conj()) / a_energy[:, None, None])

    Zt = np.einsum("fij,fjt->fit", B, Xt)
    Rv = np.einsum("fit,fjt->fij", Zt, Zt.conj()) / n_frames
    Rv += DIAG_LOAD * np.eye(n_mics)[None, :, :]

    Rinv   = np.linalg.inv(Rv)
    Rinv_a = np.einsum("fij,fj->fi", Rinv, a)
    denom  = np.einsum("fi,fi->f", a.conj(), Rinv_a)
    w      = Rinv_a / (denom[:, None] + 1e-12)

    Y = np.einsum("fi,fit->ft", w.conj(), Xt)
    _, out = istft(Y, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    N      = len(channels[0])
    out    = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
    return out.real.astype(np.float64)


# ── Spatial helpers for ILRMA DoA ─────────────────────────────────────────────
def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def predicted_tdoa(ch_a, ch_b, phi_rad):
    return (
        (MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]) * np.sin(phi_rad)
        + (MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]) * np.cos(phi_rad)
    ) / C


def srp_phat_best_angle(channels, sr):
    n     = min(len(ch) for ch in channels)
    n_fft = 1 << (n - 1).bit_length()
    chs   = [ch[:n] for ch in channels]
    gcc_s = {(a, b): gcc_phat(chs[a], chs[b], n_fft) for a, b in ALL_PAIRS}
    azs   = np.linspace(0, 360, 720, endpoint=False)
    power = np.zeros(len(azs))
    for i, az in enumerate(azs):
        phi = np.deg2rad(az); s = 0.0
        for ca, cb in ALL_PAIRS:
            tau     = predicted_tdoa(ca, cb, phi)
            lag_idx = tau * sr + n_fft // 2
            lo = int(np.floor(lag_idx)); hi = lo + 1; frac = lag_idx - lo
            lo = np.clip(lo, 0, n_fft - 1); hi = np.clip(hi, 0, n_fft - 1)
            s += (1-frac)*gcc_s[(ca, cb)][lo] + frac*gcc_s[(ca, cb)][hi]
        power[i] = s
    return float(azs[np.argmax(power)])


def steering_vector(freqs_hz, doa_deg):
    phi = np.deg2rad(doa_deg)
    tau = -(MIC_POS[:, 0]*np.sin(phi) + MIC_POS[:, 1]*np.cos(phi)) / C
    return np.exp(1j * 2*np.pi * freqs_hz[:, None] * tau[None, :])


# ── Load audio ────────────────────────────────────────────────────────────────
print(f"Loading {os.path.relpath(WAV_PATH)} ...")
sr, data = load_wav(WAV_PATH)
print(f"  {data.shape[0]} samples | {data.shape[1]} ch | {sr} Hz | "
      f"{data.shape[0]/sr:.1f} s\n")

channels = [data[:, ch] for ch in range(4)]

# Reference
print("  reference mix")
save_wav(os.path.join(OUT_DIR, "reference_mix.wav"),
         np.mean(np.stack(channels), axis=0), sr)

# Raw mics
print("\n  Raw microphone channels")
for ch, label in enumerate(MIC_LABELS):
    save_wav(os.path.join(OUT_DIR, f"mic_{label}.wav"), channels[ch], sr)

# ── Beamforming at estimated angles ───────────────────────────────────────────
print("\n  Beamforming at estimated angles ...")
for az_deg in ESTIMATED_ANGLES:
    phi = np.deg2rad(az_deg)
    lbl = angle_label(az_deg)
    print(f"  {az_deg:.0f}°")
    save_wav(os.path.join(OUT_DIR, f"das_{lbl}.wav"),
             delay_and_sum(channels, phi, sr), sr)
    save_wav(os.path.join(OUT_DIR, f"mvdr_{lbl}.wav"),
             mvdr_blocking(channels, phi, sr), sr)

# ── AuxIVA ────────────────────────────────────────────────────────────────────
for variant in AUXIVA_VARIANTS:
    print(
        f"\n  {variant['title']} (iter={variant['iters']}, STFT={variant['stft']}, "
        f"hop={variant['hop']}; {variant['note']}) ..."
    )
    remove_stale_outputs(f"{variant['prefix']}_source_*.wav")
    analysis_win_a  = pra.hann(variant["stft"])
    synthesis_win_a = pra.transform.stft.compute_synthesis_window(analysis_win_a, variant["hop"])
    X_a = pra.transform.stft.analysis(data, variant["stft"], variant["hop"], win=analysis_win_a)
    Y_a, W_a = pra.bss.auxiva(
        X_a,
        n_src=4,
        n_iter=variant["iters"],
        proj_back=False,
        return_filters=True,
    )
    gains_a = pra.bss.projection_back(Y_a, X_a.mean(axis=2))
    Y_a_mono = Y_a * gains_a[None, :, :]
    S_a = pra.transform.stft.synthesis(
        Y_a_mono,
        variant["stft"],
        variant["hop"],
        win=synthesis_win_a,
    )[:data.shape[0], :].real
    A_a = np.linalg.inv(W_a)

    for k in range(4):
        img_stft = Y_a[:, :, k][:, :, None] * A_a[:, :, k][None, :, :]
        img_chs  = [
            pra.transform.stft.synthesis(
                img_stft[:, :, m],
                variant["stft"],
                variant["hop"],
                win=synthesis_win_a,
            )[:data.shape[0]].real
            for m in range(4)
        ]
        doa  = srp_phat_best_angle(img_chs, sr)
        lbl  = angle_label(doa)
        path = os.path.join(OUT_DIR, f"{variant['prefix']}_source_{k+1}_{lbl}.wav")
        save_wav(path, S_a[:, k], sr)

# ── ILRMA ─────────────────────────────────────────────────────────────────────
mmvdr_state = None
for variant in ILRMA_VARIANTS:
    print(
        f"\n  {variant['title']} (iter={variant['iters']}, NMF_k={variant['n_components']}, "
        f"STFT={variant['stft']}, hop={variant['hop']}; {variant['note']}) ..."
    )
    remove_stale_outputs(f"{variant['prefix']}_source_*.wav")
    analysis_win_i = pra.hann(variant["stft"])
    synthesis_win_i = pra.transform.stft.compute_synthesis_window(
        analysis_win_i, variant["hop"]
    )
    X_i = pra.transform.stft.analysis(data, variant["stft"], variant["hop"], win=analysis_win_i)
    Y_i, W_i = pra.bss.ilrma(
        X_i,
        n_src=4,
        n_iter=variant["iters"],
        proj_back=False,
        n_components=variant["n_components"],
        return_filters=True,
    )
    gains_i = pra.bss.projection_back(Y_i, X_i.mean(axis=2))
    Y_i_mono = Y_i * gains_i[None, :, :]
    S_i = pra.transform.stft.synthesis(
        Y_i_mono,
        variant["stft"],
        variant["hop"],
        win=synthesis_win_i,
    )[:data.shape[0], :].real
    A_i = np.linalg.inv(W_i.transpose(0, 2, 1)).transpose(0, 2, 1)

    for k in range(4):
        img_stft = Y_i[:, :, k][:, :, None] * A_i[:, :, k][None, :, :]
        img_chs = [
            pra.transform.stft.synthesis(
                img_stft[:, :, m],
                variant["stft"],
                variant["hop"],
                win=synthesis_win_i,
            )[:data.shape[0]].real
            for m in range(4)
        ]
        doa = srp_phat_best_angle(img_chs, sr)
        lbl = angle_label(doa)
        path = os.path.join(OUT_DIR, f"{variant['prefix']}_source_{k+1}_{lbl}.wav")
        save_wav(path, S_i[:, k], sr)

    if variant["key"] == "default":
        mmvdr_state = {
            "stft": variant["stft"],
            "hop": variant["hop"],
            "X": X_i,
            "Y": Y_i,
            "A": A_i,
            "synthesis_win": synthesis_win_i,
        }

# ── Mask-MVDR ─────────────────────────────────────────────────────────────────
if mmvdr_state is None:
    raise RuntimeError("Default ILRMA state missing for Mask-MVDR generation.")

print(f"\n  Mask-MVDR using default ILRMA masks ...")
remove_stale_outputs("mmvdr_source_*.wav")
ILRMA_STFT = mmvdr_state["stft"]
ILRMA_HOP = mmvdr_state["hop"]
X_i = mmvdr_state["X"]
Y_i = mmvdr_state["Y"]
A_i = mmvdr_state["A"]
synthesis_win_i = mmvdr_state["synthesis_win"]
freqs_hz = np.fft.rfftfreq(ILRMA_STFT, 1.0 / sr)
X_fmt    = X_i.transpose(1, 2, 0)                                # (F, M, T)
Y_power  = np.abs(Y_i) ** 2
masks    = Y_power / (Y_power.sum(axis=2, keepdims=True) + 1e-12)  # (T, F, nsrc)

for k in range(4):
    # DoA from ILRMA image
    img_stft = Y_i[:, :, k][:, :, None] * A_i[:, :, k][None, :, :]
    img_chs  = [pra.transform.stft.synthesis(img_stft[:, :, m], ILRMA_STFT, ILRMA_HOP,
                                              win=synthesis_win_i)[:data.shape[0]].real
                for m in range(4)]
    doa = srp_phat_best_angle(img_chs, sr)
    lbl = angle_label(doa)

    a   = steering_vector(freqs_hz, doa)       # (F, M)
    mk  = masks[:, :, k].T                     # (F, T)
    mi  = 1.0 - mk

    Rk  = np.einsum("ft,fmi,fni->fmn", mk, X_fmt, X_fmt.conj()) / X_fmt.shape[2]
    Rv  = np.einsum("ft,fmi,fni->fmn", mi, X_fmt, X_fmt.conj()) / X_fmt.shape[2]
    Rv += DIAG_LOAD * np.eye(4)[None, :, :]

    Rinv   = np.linalg.inv(Rv)
    Rinv_a = np.einsum("fmn,fn->fm", Rinv, a)
    denom  = np.einsum("fm,fm->f", a.conj(), Rinv_a).real + 1e-12
    w      = Rinv_a / denom[:, None]

    Y_beam = np.einsum("fm,fmt->ft", w.conj(), X_fmt)
    _, out = istft(Y_beam, fs=sr, nperseg=ILRMA_STFT, noverlap=ILRMA_STFT - ILRMA_HOP)
    N  = data.shape[0]
    out = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
    save_wav(os.path.join(OUT_DIR, f"mmvdr_source_{k+1}_{lbl}.wav"),
             out.real.astype(np.float64), sr)

print(f"\nDone — all files written to {os.path.relpath(OUT_DIR)}/")
