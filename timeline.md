# Project Timeline

## Software Case — Speaker Separation

### Session 1

- Read all 5 case PDFs (DSP, Electrical, Embedded, Mechanical, Software)
- Decided to focus on **Software Case** (multi-talker extraction) and **DSP Case**
- Looked up Audio Explorers 2026 competition details (Demant, prize trip to Toronto, deadline April 12)
- Explored repo structure; data in `data/`, originals in `DONT-TOUCH/`
- Plotted raw spectrogram of `mixture.wav` — 21s, 4-channel, 44.1kHz, speech energy 0–4kHz

### Beamforming exploration

- **gcc-das-01**: GCC-PHAT calibration from `example_mixture.wav` + integer delay-and-sum.
  Failed — 4 simultaneous sources corrupt GCC peaks, most steering delays wrong.
  Key finding: τ_lr = 29 samples (0.66ms) is reliable; τ_fr = 8 samples less so.

- **geom-das-01**: Geometry model (mic positions from τ_lr/τ_fr) + frequency-domain
  beamforming with fractional delays. Beam patterns look correct. Spectrograms
  still near-identical — fundamental aperture ceiling (~6dB suppression).

- **mvdr-01**: Geometry-based broadband MVDR with loaded full-mixture covariance.
  Stable numerically (median loaded condition number ~86), and beam patterns show
  deeper, adaptive nulls than DAS, especially for the front/back beams.
  Still not clean separation — side-looking beams are asymmetric and there is
  substantial bleed because covariance is estimated from the same 4-speaker mix.

- **example-validation-01**: Reframed discovery properly around `example_mixture.wav`
  only. Compared closest-mic baselines vs geometry-DAS vs geometry-MVDR on the
  known 0° / 90° / 180° / 270° scene. Result: geometry-DAS does not clearly beat
  the closest mic; MVDR improves cross-beam differentiation somewhat, but still
  leaves strong front/back confusion and does not cleanly isolate talkers.

### Up next

- Keep discovery and transfer separate:
  validate only on `example_mixture.wav`, then later estimate DoA for
  `mixture.wav` with a separate localization step
