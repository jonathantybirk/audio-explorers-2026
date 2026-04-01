"""
Interactive beamforming demo.
Run with:  streamlit run demo/beamforming_demo.py

Features:
  - SRP-PHAT polar plot (live, updates with mic pair selection)
  - Delay-and-Sum beamformer
  - MVDR beamformer
  - Audio playback of the beamformed output
"""

import io
import os
import json
import itertools

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import stft, istft

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WAV_FILES = {
    "example_mixture (known positions)": os.path.join(
        REPO_ROOT, "DONT-TOUCH/Software Case/example_mixture.wav"
    ),
    "mixture (unknown positions)": os.path.join(
        REPO_ROOT, "DONT-TOUCH/Software Case/mixture.wav"
    ),
}
GEO_PATH = os.path.join(REPO_ROOT, "data", "mic_geometry.json")

LF, LR, RF, RR = 0, 1, 2, 3
CHANNEL_NAMES = {LF: "LF", LR: "LR", RF: "RF", RR: "RR"}
ALL_PAIRS = list(itertools.combinations([LF, LR, RF, RR], 2))
PAIR_LABELS = {
    (LF, RF): "LF–RF  (inter-ear, left/right)",
    (LR, RR): "LR–RR  (inter-ear rear, left/right)",
    (LF, LR): "LF–LR  (intra-ear left, front/back)",
    (RF, RR): "RF–RR  (intra-ear right, front/back)",
    (LF, RR): "LF–RR  (diagonal)",
    (LR, RF): "LR–RF  (diagonal)",
}


# ── Load geometry ─────────────────────────────────────────────────────────────
with open(GEO_PATH) as f:
    geo = json.load(f)
D = geo["inter_ear_distance_m"]
L = geo["intra_ear_spacing_m"]
C = geo["speed_of_sound_m_s"]

MIC_POS = {
    LF: np.array([ D/2,  L/2]),
    LR: np.array([ D/2, -L/2]),
    RF: np.array([-D/2,  L/2]),
    RR: np.array([-D/2, -L/2]),
}


# ── Signal processing helpers ─────────────────────────────────────────────────
def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    else:
        data = data.astype(np.float32)
    return sr, data


def gcc_phat(x, y, n_fft):
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    G = X * np.conj(Y)
    G /= np.abs(G) + 1e-12
    return np.fft.fftshift(np.fft.irfft(G, n=n_fft))


def tdoa(ch_a, ch_b, phi_rad):
    pa, pb = MIC_POS[ch_a], MIC_POS[ch_b]
    return ((pa[0] - pb[0]) * np.sin(phi_rad) +
            (pa[1] - pb[1]) * np.cos(phi_rad)) / C


def srp_phat(gcc_store, pairs, azimuths_deg, sr, n_fft):
    power = np.zeros(len(azimuths_deg))
    for k, az in enumerate(azimuths_deg):
        phi = np.deg2rad(az)
        s = 0.0
        for ch_a, ch_b in pairs:
            tau     = tdoa(ch_a, ch_b, phi)
            lag_idx = tau * sr + n_fft // 2
            lo      = int(np.clip(np.floor(lag_idx), 0, n_fft - 1))
            hi      = int(np.clip(lo + 1,            0, n_fft - 1))
            frac    = lag_idx - np.floor(lag_idx)
            gcc     = gcc_store[(ch_a, ch_b)]
            s += (1 - frac) * gcc[lo] + frac * gcc[hi]
        power[k] = s
    return power


def steering_vector(freqs, phi_rad):
    """Shape: (n_freqs, 4)"""
    delays = np.array([
        -(MIC_POS[m][0] * np.sin(phi_rad) + MIC_POS[m][1] * np.cos(phi_rad)) / C
        for m in [LF, LR, RF, RR]
    ])
    return np.exp(1j * 2 * np.pi * freqs[:, None] * delays[None, :])


def beamform_das(data_stft, freqs, phi_rad):
    """Delay-and-Sum: shape data_stft = (4, n_freqs, n_frames)"""
    sv = steering_vector(freqs, phi_rad)          # (n_freqs, 4)
    weighted = data_stft * sv.T[:, :, None]       # (4, n_freqs, n_frames)
    return weighted.mean(axis=0)                  # (n_freqs, n_frames)


def beamform_mvdr(data_stft, freqs, phi_rad, diag_load=1e-3):
    """
    MVDR: estimate spatial covariance from the mixture, then apply
    distortionless constraint toward phi_rad.
    data_stft shape: (4, n_freqs, n_frames)
    """
    n_mics, n_freqs, n_frames = data_stft.shape
    output = np.zeros((n_freqs, n_frames), dtype=complex)

    sv = steering_vector(freqs, phi_rad)  # (n_freqs, 4)

    for fi in range(n_freqs):
        X = data_stft[:, fi, :]                        # (4, n_frames)
        R = (X @ X.conj().T) / n_frames                # (4, 4) covariance
        R += diag_load * np.eye(n_mics)                # regularise
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.eye(n_mics)
        a = sv[fi]                                     # (4,)
        denom = a.conj() @ Rinv @ a
        if abs(denom) < 1e-12:
            w = np.ones(n_mics) / n_mics
        else:
            w = (Rinv @ a) / denom                     # (4,) MVDR weights
        output[fi, :] = w.conj() @ X

    return output  # (n_freqs, n_frames)


def array_to_wav_bytes(signal, sr):
    sig = np.clip(signal / (np.max(np.abs(signal)) + 1e-9), -1, 1)
    sig = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, sig)
    return buf.getvalue()


# ── Heavy precomputation (cached per file) ────────────────────────────────────
@st.cache_data(show_spinner="Loading audio and computing GCC-PHAT…")
def precompute(wav_path):
    sr, data = load_wav(wav_path)
    n        = data.shape[0]
    n_fft_gcc = 1 << (n - 1).bit_length()
    lags     = (np.arange(n_fft_gcc) - n_fft_gcc // 2) / sr
    gcc_store = {
        (a, b): gcc_phat(data[:, a].astype(np.float64),
                         data[:, b].astype(np.float64), n_fft_gcc)
        for a, b in ALL_PAIRS
    }

    # STFT for beamforming
    nperseg = 512
    _, _, Zxx = stft(data[:, 0], fs=sr, nperseg=nperseg)
    n_freqs, n_frames = Zxx.shape
    freqs = np.fft.rfftfreq(nperseg, 1 / sr)
    data_stft = np.zeros((4, n_freqs, n_frames), dtype=complex)
    for ch in range(4):
        _, _, data_stft[ch] = stft(data[:, ch], fs=sr, nperseg=nperseg)

    return sr, data, gcc_store, n_fft_gcc, data_stft, freqs, nperseg


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Beamforming Demo", layout="wide")
st.title("Hearing Aid Beamforming Demo")
st.caption(f"Mic geometry — inter-ear D = {D*100:.1f} cm, intra-ear L = {L*100:.1f} cm")

col_left, col_right = st.columns([1, 2])

with col_left:
    wav_label = st.radio("Recording", list(WAV_FILES.keys()))
    wav_path  = WAV_FILES[wav_label]

    st.markdown("**Mic pairs for SRP-PHAT**")
    selected_pairs = []
    for ch_a, ch_b in ALL_PAIRS:
        label = PAIR_LABELS[(ch_a, ch_b)]
        if st.checkbox(label, value=True):
            selected_pairs.append((ch_a, ch_b))

    st.divider()
    st.markdown("**Beamformer**")
    bf_method = st.radio("Method", ["Delay-and-Sum", "MVDR"])
    az_target = st.slider("Steer direction (°)", 0, 359, 0, step=1,
                          help="0=front, 90=left, 180=back, 270=right")
    run_bf = st.button("Beamform & play", type="primary")

# ── Load & precompute ─────────────────────────────────────────────────────────
sr, data, gcc_store, n_fft_gcc, data_stft, freqs, nperseg = precompute(wav_path)

# ── SRP-PHAT polar plot ───────────────────────────────────────────────────────
with col_right:
    st.subheader("SRP-PHAT spatial spectrum")

    if not selected_pairs:
        st.warning("Select at least one mic pair.")
    else:
        azimuths = np.linspace(0, 360, 720, endpoint=False)
        power    = srp_phat(gcc_store, selected_pairs, azimuths, sr, n_fft_gcc)
        power_n  = (power - power.min()) / (power.max() - power.min() + 1e-12)

        # Plotly polar: convention 0°=front=top, clockwise
        plot_az = (90 - azimuths) % 360  # convert to standard math polar for plotly

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=power_n,
            theta=plot_az,
            mode="lines",
            fill="toself",
            fillcolor="rgba(70,130,180,0.2)",
            line=dict(color="steelblue", width=1.5),
            name="SRP-PHAT power",
        ))
        # Steering direction marker
        fig.add_trace(go.Scatterpolar(
            r=[0, 1.15],
            theta=[(90 - az_target) % 360] * 2,
            mode="lines",
            line=dict(color="crimson", width=2, dash="dash"),
            name=f"Steer {az_target}°",
        ))
        if "example" in wav_label:
            for az, lbl in [(0, "front"), (90, "left"), (180, "back"), (270, "right")]:
                pa = (90 - az) % 360
                fig.add_trace(go.Scatterpolar(
                    r=[1.05], theta=[pa], mode="markers+text",
                    marker=dict(color="orange", size=10, symbol="diamond"),
                    text=[lbl], textposition="top center",
                    textfont=dict(color="orange", size=10),
                    showlegend=False,
                ))
        fig.update_layout(
            polar=dict(
                angularaxis=dict(
                    tickmode="array",
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=["E(270°)", "NE", "N(0°/front)", "NW", "W(90°/left)", "SW", "S(180°/back)", "SE"],
                    direction="clockwise",
                    rotation=90,
                ),
                radialaxis=dict(visible=False),
            ),
            showlegend=True,
            height=480,
            margin=dict(l=40, r=40, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Beamforming ───────────────────────────────────────────────────────────
    if run_bf:
        phi = np.deg2rad(az_target)
        with st.spinner(f"Running {bf_method} toward {az_target}°…"):
            if bf_method == "Delay-and-Sum":
                bf_stft = beamform_das(data_stft, freqs, phi)
            else:
                bf_stft = beamform_mvdr(data_stft, freqs, phi)

            _, bf_signal = istft(bf_stft, fs=sr, nperseg=nperseg)
            bf_signal = bf_signal.real.astype(np.float32)

        st.markdown(f"**{bf_method} output steered to {az_target}°**")
        wav_bytes = array_to_wav_bytes(bf_signal, sr)
        st.audio(wav_bytes, format="audio/wav")

    st.caption(
        "Polar plot: 0°/front = top, 90°/left = left, clockwise. "
        "Orange diamonds = known talker positions (example_mixture only). "
        "Red dashed line = current beam direction."
    )
