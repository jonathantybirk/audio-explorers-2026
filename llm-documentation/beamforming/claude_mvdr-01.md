# mvdr-01 -- Geometry-Based Broadband MVDR

## What this attempt does

Keeps the same explicit 4-mic geometry from `geom-das-01`, but replaces equal
delay-and-sum weights with broadband MVDR weights estimated in the STFT domain.

For each steering direction, the script:

1. Computes the geometry-based steering vector `d(f)` from the trusted delays
   `tau_lr = 29` samples and `tau_fr = 8` samples.
2. Estimates one spatial covariance matrix `R(f)` per frequency bin directly
   from `mixture.wav`.
3. Applies diagonal loading for stability:

   `R_loaded(f) = R(f) + 0.01 * trace(R(f)) / M * I`

4. Solves the MVDR weights:

   `w(f) = R_loaded(f)^-1 d(f) / (d(f)^H R_loaded(f)^-1 d(f))`

5. Reconstructs one broadband output for each cardinal steering direction.

## Parameters used

- STFT: Hann window, `nperseg = 1024`, `noverlap = 768`
- Diagonal loading: `0.01 * trace(R) / M`
- Beam-pattern summary frequencies: `500, 1000, 2000, 3500 Hz`

## What worked

The implementation is numerically stable:

- Loaded covariance condition number is moderate:
  median `~86`, 95th percentile `~401`
- The distortionless constraint is satisfied to machine precision:
  mean `|w^H d - 1|` is about `1e-16` for all 4 beams

The beam patterns are more selective than plain DAS, especially for the
front/back beams:

- `0deg_front` response to `90deg_right`: `-16.96 dB`
- `180deg_back` response to `90deg_right`: `-16.36 dB`

That is noticeably deeper than the roughly `-9 dB` side suppression seen in
the frequency-averaged DAS patterns.

## What did not improve enough

This is still not clean speaker separation.

- The `90deg_right` and `270deg_left` beams are visibly asymmetric.
- Their response is still partly pulled toward the dominant content in the
  mixture, instead of forming clean nulls against all competing directions.
- The differential spectrograms show stronger directional differences than DAS,
  but there is still substantial bleed across all four outputs.

So MVDR helped, but it did not solve the case by itself.

## Why the limitation remains

The covariance is estimated from the same fully overlapped 4-speaker mixture
that the beamformer is trying to separate. That means:

- target and interferers are mixed into the same `R(f)`
- there is no clean noise-only or interference-only covariance
- the dominant speakers can bias the adaptive weights

The physical constraints also still matter:

- front/back aperture is small because `tau_fr = 8` samples
- low frequencies remain weakly separable for any spatial method

## Takeaway

`mvdr-01` is a valid next non-deep baseline after DAS. It improves directional
selectivity and produces more differentiated outputs, but the current blind,
global covariance estimate is not strong enough for clean multi-talker
extraction in this scene.

## Most sensible next non-deep step

Refine MVDR rather than abandoning it immediately:

- use time-local or recursively updated covariance instead of one global `R(f)`
- add a postfilter / Wiener-style suppression stage after the beamformer
- keep objective comparison against `geom-das-01` instead of relying only on
  spectrogram inspection
