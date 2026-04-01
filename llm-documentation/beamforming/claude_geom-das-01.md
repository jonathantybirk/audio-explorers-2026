# geom-das-01 — Geometry-Based Frequency-Domain Delay-and-Sum

## Improvements over gcc-das-01

Two concrete changes:

1. **Geometry model instead of raw GCC peaks** — mic positions placed explicitly
   in 2D using the two trusted measurements from gcc-das-01. Steering delays
   computed analytically via the far-field formula rather than trying to read
   noisy peaks from a 4-source simultaneous GCC-PHAT.

2. **Frequency-domain beamforming with fractional delays** — instead of integer
   `np.roll` shifts, each channel gets an exact phase rotation per FFT bin:
   `Y[k] = (1/M) Σ_i X_i[k] · exp(+j·2π·k·τ_i / N)`
   This handles sub-sample delays correctly, which matters especially for
   the small τ_fr delays.

## Mic layout

```
x = right (+), y = front (+)

LF: (-τ_lr/2,  +τ_fr/2)    LR: (-τ_lr/2,  -τ_fr/2)
RF: (+τ_lr/2,  +τ_fr/2)    RR: (+τ_lr/2,  -τ_fr/2)

τ_lr = 29 samples (0.66ms) — high confidence from GCC peak
τ_fr =  8 samples (0.18ms) — moderate confidence, likely slight underestimate
```

## Computed steering delays

| Direction  | τ_LF | τ_LR    | τ_RF    | τ_RR    |
|------------|------|---------|---------|---------|
| 0° front   | 0    | +8      | 0       | +8      |
| 90° right  | 0    | 0       | −29     | −29     |
| 180° back  | 0    | −8      | 0       | −8      |
| 270° left  | 0    | 0       | +29     | +29     |

These are now physically consistent across all 4 mics — the main bug in
gcc-das-01 (90° only got −5 samples instead of −29) is fixed.

## Results

**Beam patterns** look correct. The 90° and 270° beams show clear directional
lobes at the right angles with proper side-lobe structure. The 0° and 180°
beams are visibly weaker (broader, less suppression) because τ_fr=8 samples
gives much less directivity than τ_lr=29 samples. This is physically expected.

**Spectrograms** still look very similar across beams. This is fundamental,
not a bug:
- With only 4 mics and a small aperture (~22cm inter-ear), maximum suppression
  of off-axis sources is ~6dB at mid-frequencies.
- At low frequencies (< ~750Hz), the wavelength exceeds the aperture and
  spatial filtering breaks down entirely.
- 4 simultaneous talkers with ~equal energy means even perfect 6dB suppression
  leaves a lot of bleed.

## Key limitations of delay-and-sum

- Suppression is proportional to aperture / wavelength. At 1kHz the aperture
  (22cm) is only ~0.65 wavelengths — barely enough for modest directivity.
- DAS weights all mics equally regardless of direction. An adaptive beamformer
  (MVDR/Wiener) would optimise weights to null specific interference directions
  rather than just steer toward the target.
- The geometry model assumes perfect far-field point sources. Real sources at
  1.9m are borderline near-field for a 22cm aperture (far-field condition is
  typically d >> aperture²/λ).

## What to try next

- **MVDR beamformer**: estimate the spatial covariance matrix from the signal,
  then solve for weights that minimise total output power while preserving
  gain toward the steering direction. Should give 10–20dB suppression vs 3–6dB.
- **Deep learning source separation**: bypass the aperture problem entirely.
  Models like SepFormer or Conv-TasNet trained on multi-speaker data can
  exploit spectral patterns that are invisible to spatial methods.
