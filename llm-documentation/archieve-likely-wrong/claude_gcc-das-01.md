# gcc-das-01 — GCC-PHAT Calibrated Delay-and-Sum

## What this attempt does

Uses `example_mixture.wav` (4 known sources at 0°, 90°, 180°, 270°) to empirically
measure inter-mic TDOAs via GCC-PHAT, then applies integer delay-and-sum beamforming
to `mixture.wav` steered toward each of the 4 known directions.

No physical mic distances are assumed — delays are read directly from GCC-PHAT peaks,
using only the channel labels (LF/LR/RF/RR) to assign which peak belongs to which direction.

## GCC-PHAT strategy

Three pairs computed relative to LF:

- **GCC(LF, RF)** — left/right sensitive: front/back sources land near zero lag,
  giving clean off-centre peaks for 90° and 270°.
- **GCC(LF, LR)** — front/rear sensitive: side sources land near zero lag,
  giving clean off-centre peaks for 0° and 180°.
- **GCC(LF, RR)** — diagonal pair: all 4 sources give distinct peaks.
  Assigned by seeding a small search window around the τ values found
  in the above two pairs (RR shares rear behaviour with LR and right
  behaviour with RF).

## Measured steering delays

| Direction  | τ_LF | τ_LR       | τ_RF       | τ_RR       |
|------------|------|------------|------------|------------|
| 0° front   | 0    | +8 (+0.18ms) | 0        | 0          |
| 90° right  | 0    | 0          | -5 (-0.11ms) | 0        |
| 180° back  | 0    | -6 (-0.14ms) | 0        | 0          |
| 270° left  | 0    | 0          | +29 (+0.66ms) | +29 (+0.66ms) |

## What went wrong

**The GCC-PHAT of a 4-source simultaneous mixture is unreliable for isolating individual source TDOAs.**

- 270° front is the dominant source in the recording and produces a clean, large peak at
  +29 samples in GCC(LF, RF). All other directions have their peaks buried or contaminated.
- 90° right should give a symmetric -29 sample peak in GCC(LF, RF), but the negative side
  shows only a small peak at -5 samples — almost certainly contamination from the zero-lag
  region rather than a real 90° measurement.
- The RR diagonal pair mostly snapped to 0 because the seeded windows found the zero-lag
  peak rather than real off-centre peaks.
- As a result, almost all directions got near-zero delays and the 4 beamformed outputs
  look nearly identical.

## Key measurements to carry forward

- **τ_lr = 29 samples (0.66ms)** — high confidence. This is the cleanest, most prominent
  peak in the entire analysis. Corresponds to an inter-ear mic distance of ~22.5cm
  (plausible for hearing aids sitting on the outer ear). Use as ground truth.
- **τ_fr = 8 samples (0.18ms)** — moderate confidence. Close to the zero-lag region,
  likely slightly underestimated. Corresponds to ~6.2mm front-rear mic spacing
  (typical Oticon hearing aid is 10–15mm). Treat as a lower-bound estimate.

## What to try next (geom-das-01)

1. **Geometry model** — place all 4 mics explicitly using the measured τ_lr (trusted)
   and τ_fr (ballpark). Compute theoretical far-field TDOAs for all directions via
   `τ = (d · cos(θ - φ)) / c`. This avoids relying on the noisy multi-source GCC peaks
   and gives consistent delays across all 4 mics.

2. **Frequency-domain beamforming with fractional delays** — integer shifts (used here)
   introduce phase errors that worsen at higher frequencies. Applying exact phase rotations
   per FFT bin (`e^{-j2πf·τ}`) handles sub-sample precision correctly.
