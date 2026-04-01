# Idea: MVDR Beamforming

## The idea

MVDR (Minimum Variance Distortionless Response) is the natural next step after
delay-and-sum. Instead of equal weights across all mics, it solves for the
weight vector w that minimises total output power subject to the constraint
that the response toward the steering direction is preserved (gain = 1).

    w_MVDR = R^{-1} · d / (d^H · R^{-1} · d)

where:
- R  = spatial covariance matrix estimated from the signal (n_mics × n_mics)
- d  = steering vector (phase shifts for the target direction)

Because it actively minimises power from all other directions, it can
theoretically null out specific interference sources — giving 10–20dB
suppression vs the 3–6dB ceiling of DAS.

## Why this is promising here

- We already have the steering vectors from geom-das-01.
- The covariance matrix R can be estimated directly from mixture.wav.
- We have 4 mics, so R is 4×4 — cheap to invert.
- Works per frequency bin in the STFT domain (broadband MVDR).

## Concerns / things to watch

- R estimation requires enough data and ideally noise-only segments to
  separate signal from noise covariance. With 4 simultaneously active
  talkers there is no clean noise reference — R will include all sources.
- If the target source dominates R, the MVDR solution degrades toward DAS.
- May need diagonal loading (R + εI) for numerical stability.
- Still limited by aperture at low frequencies — same physical constraint as DAS.
