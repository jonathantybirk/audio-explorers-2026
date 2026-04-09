"""
Steered beamforming for mixture.wav.

Pipeline:
  1. Dense SRP-PHAT scan → find azimuth peaks (speaker directions)
  2. For each peak, steer a Delay-and-Sum beamformer (phase-aligned sum)
     AND an MVDR beamformer (covariance-based null steering)
  3. Save one WAV per direction per method

Outputs: analysis/steered_beamform/outputs/
"""

import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, istft, stft

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAV  = os.path.join(REPO, "DONT-TOUCH/Software Case/mixture.wav")
GEO  = os.path.join(REPO, "data/mic_geometry.json")
OUT  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

STFT_SIZE = 2048
HOP_SIZE  = 512
AZ_STEP   = 0.5          # degrees per SRP-PHAT grid point
PEAK_MIN_DIST_DEG = 25   # minimum separation between peaks (degrees)
PEAK_PROMINENCE   = 0.04  # relative to max power
N_PEAKS_MAX       = 9    # never report more than this

with open(GEO) as f:
    geo = json.load(f)
D  = geo["inter_ear_distance_m"]
L  = geo["intra_ear_spacing_m"]
C  = geo["speed_of_sound_m_s"]

# [LF, LR, RF, RR] in (x, y) — x=left-right, y=front-back
MIC_POS = np.array([
    [ D/2,  L/2],   # LF
    [ D/2, -L/2],   # LR
    [-D/2,  L/2],   # RF
    [-D/2, -L/2],   # RR
], dtype=np.float64)

ALL_PAIRS = list(itertools.combinations(range(4), 2))


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2**31
    else:
        data = data.astype(np.float64)
    return sr, data


def save_wav(path, sig, sr):
    peak = np.max(np.abs(sig)) + 1e-12
    out = np.clip(sig / peak * 0.9, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  saved {os.path.relpath(path)}", flush=True)


# ── Geometry ──────────────────────────────────────────────────────────────────

def steering_delay(az_deg):
    """Return per-mic time delays (seconds) for a plane wave from az_deg."""
    phi = np.deg2rad(az_deg)
    # unit vector pointing toward source (azimuth from front, clockwise)
    ux = np.sin(phi)   # left component
    uy = np.cos(phi)   # front component
    # positive delay = sound arrives later at this mic
    delays = -(MIC_POS[:, 0] * ux + MIC_POS[:, 1] * uy) / C
    delays -= delays.min()
    return delays


def steering_vector(az_deg, freqs):
    """Complex steering vector a(az, f): shape (n_mics, n_freqs)."""
    delays = steering_delay(az_deg)       # (4,)
    # a[m, f] = exp(-j * 2pi * f * tau_m)
    return np.exp(-1j * 2 * np.pi * freqs[None, :] * delays[:, None])


# ── SRP-PHAT ──────────────────────────────────────────────────────────────────

def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def srp_phat_scan(channels, sr, azimuths):
    n = min(len(ch) for ch in channels)
    channels = [ch[:n] for ch in channels]
    n_fft = 1 << (n - 1).bit_length()
    gcc_store = {(a, b): gcc_phat(channels[a], channels[b], n_fft)
                 for a, b in ALL_PAIRS}
    power = np.zeros(len(azimuths))
    for idx, az in enumerate(azimuths):
        phi = np.deg2rad(az)
        score = 0.0
        for ch_a, ch_b in ALL_PAIRS:
            tau = (  (MIC_POS[ch_a, 0] - MIC_POS[ch_b, 0]) * np.sin(phi)
                   + (MIC_POS[ch_a, 1] - MIC_POS[ch_b, 1]) * np.cos(phi)) / C
            lag = tau * sr + n_fft // 2
            lo = int(np.floor(lag)); hi = lo + 1; frac = lag - lo
            lo = np.clip(lo, 0, n_fft - 1); hi = np.clip(hi, 0, n_fft - 1)
            gcc = gcc_store[(ch_a, ch_b)]
            score += (1 - frac) * gcc[lo] + frac * gcc[hi]
        power[idx] = score
    return power


def find_srp_peaks(power, azimuths, min_dist_deg, prominence_rel):
    az_step = azimuths[1] - azimuths[0]
    min_dist_samples = max(1, int(min_dist_deg / az_step))
    prom = (power.max() - power.min()) * prominence_rel
    # Wrap-around: duplicate the array to handle 0°/360° boundary
    power2 = np.concatenate([power, power])
    az2    = np.concatenate([azimuths, azimuths + 360])
    peaks2, props = find_peaks(power2, distance=min_dist_samples, prominence=prom)
    # Keep only peaks in first copy, unwrap
    peaks2 = peaks2[peaks2 < len(azimuths)]
    peaks2 = peaks2[np.argsort(props["prominences"][:len(peaks2)])[::-1]]
    peaks2 = peaks2[:N_PEAKS_MAX]
    peak_azimuths = azimuths[peaks2]
    peak_powers   = power[peaks2]
    # Sort by azimuth
    order = np.argsort(peak_azimuths)
    return peak_azimuths[order], peak_powers[order]


# ── Delay-and-Sum beamformer ──────────────────────────────────────────────────

def das_beamform(data, sr, az_deg):
    """Time-domain delay-and-sum toward az_deg. Returns mono signal."""
    delays = steering_delay(az_deg)
    n = data.shape[0]
    out = np.zeros(n)
    for m in range(data.shape[1]):
        shift_samples = delays[m] * sr
        # sub-sample shift via FFT phase shift
        sig = data[:, m]
        S = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0/sr)
        S *= np.exp(-1j * 2 * np.pi * freqs * shift_samples / sr)
        out += np.fft.irfft(S, n=n).real
    return (out / data.shape[1]).astype(np.float64)


# ── MVDR beamformer ───────────────────────────────────────────────────────────

def mvdr_beamform(data, sr, az_deg):
    """
    MVDR: steer toward az_deg, compute interference covariance from
    all-but-target spatial covariance.
    """
    _, _, Zxx = stft(data.T, fs=sr, nperseg=STFT_SIZE, noverlap=STFT_SIZE - HOP_SIZE)
    # Zxx: (n_mics, n_freqs, n_frames)
    n_mics, n_freqs, n_frames = Zxx.shape
    freqs = np.fft.rfftfreq(STFT_SIZE, d=1.0/sr)

    # Full spatial covariance: (n_freqs, n_mics, n_mics)
    R = np.einsum('mft,nft->fmn', Zxx, np.conj(Zxx)) / n_frames

    a = steering_vector(az_deg, freqs)   # (n_mics, n_freqs)

    Y = np.zeros((n_freqs, n_frames), dtype=complex)
    eps = 1e-6 * np.eye(n_mics)
    for f in range(n_freqs):
        Rf = R[f] + eps
        af = a[:, f]
        try:
            Ri = np.linalg.solve(Rf, af)
            w = Ri / (af.conj() @ Ri + 1e-12)
        except np.linalg.LinAlgError:
            w = af / (n_mics)
        Y[f] = w.conj() @ Zxx[:, f, :]

    # ISTFT
    _, out = istft(Y, fs=sr, nperseg=STFT_SIZE, noverlap=STFT_SIZE - HOP_SIZE)
    n = data.shape[0]
    if len(out) > n:
        out = out[:n]
    elif len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out.real.astype(np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {os.path.relpath(WAV)} ...")
    sr, data = load_wav(WAV)
    channels = [data[:, m] for m in range(data.shape[1])]
    print(f"  {data.shape[0]/sr:.1f}s  |  {data.shape[1]} ch  |  {sr} Hz")

    # ── SRP-PHAT scan ─────────────────────────────────────────────────────────
    print("\nRunning SRP-PHAT scan ...")
    azimuths = np.arange(0, 360, AZ_STEP)
    power = srp_phat_scan(channels, sr, azimuths)

    peak_az, peak_pw = find_srp_peaks(power, azimuths, PEAK_MIN_DIST_DEG, PEAK_PROMINENCE)
    n_peaks = len(peak_az)
    print(f"\nFound {n_peaks} peaks:")
    for i, (az, pw) in enumerate(zip(peak_az, peak_pw)):
        print(f"  Peak {i+1}: {az:.1f}°  (SRP power {pw:.4f})")

    # ── Plot SRP-PHAT spectrum ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(azimuths, power, linewidth=1, color="steelblue")
    for az in peak_az:
        ax.axvline(az, color="crimson", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(az + 1, power.max() * 0.95, f"{az:.0f}°", fontsize=9, color="crimson")
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("SRP-PHAT power")
    ax.set_title(f"SRP-PHAT spectrum — mixture.wav  ({n_peaks} peaks detected)")
    ax.set_xlim(0, 360)
    plt.tight_layout()
    scan_plot = os.path.join(OUT, "srp_phat_scan.png")
    plt.savefig(scan_plot, dpi=150)
    plt.close()
    print(f"\n  saved {os.path.relpath(scan_plot)}")

    # ── Beamform toward each peak ─────────────────────────────────────────────
    print("\nBeamforming toward each peak ...")
    for i, az in enumerate(peak_az):
        label = f"peak{i+1}_{az:.0f}deg"

        print(f"\n  [{i+1}/{n_peaks}] {az:.1f}° — Delay-and-Sum ...", flush=True)
        das = das_beamform(data, sr, az)
        save_wav(os.path.join(OUT, f"das_{label}.wav"), das, sr)

        print(f"  [{i+1}/{n_peaks}] {az:.1f}° — MVDR ...", flush=True)
        mvdr = mvdr_beamform(data, sr, az)
        save_wav(os.path.join(OUT, f"mvdr_{label}.wav"), mvdr, sr)

    print(f"\nDone. {n_peaks * 2} files written to {os.path.relpath(OUT)}")
    print("\nSummary:")
    for i, az in enumerate(peak_az):
        print(f"  Source {i+1}: {az:.1f}°")


if __name__ == "__main__":
    main()
