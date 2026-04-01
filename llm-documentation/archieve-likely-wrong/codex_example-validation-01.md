# example-validation-01 -- Example-Only Validation Pack

## Intent

This validation step uses `example_mixture.wav` only.

That is deliberate: the example scene is the only recording where the source
layout is known (`0°, 90°, 180°, 270°`). It is therefore the right place to
test whether a beamforming method actually improves anything before trying to
generalise to `mixture.wav`.

## What was compared

For each known direction, three outputs were generated:

1. **Closest mic baseline**
   - `0deg_front` -> `LF`
   - `90deg_right` -> `RF`
   - `180deg_back` -> `RR`
   - `270deg_left` -> `LR`

2. **Geometry-DAS**
   - same geometry model as `geom-das-01`

3. **Geometry-MVDR**
   - same geometry model as `mvdr-01`
   - covariance estimated from `example_mixture.wav`

All outputs were saved in:

`analysis/beam-forming/example-validation-01/audio/`

## What the metrics say

### Closest mic vs geometry-DAS

Geometry-DAS does **not** clearly improve over the closest-mic baseline.

- Closest-mic average pairwise log-magnitude correlation: `0.8687`
- Geometry-DAS average pairwise log-magnitude correlation: `0.8708`

That means the geometry-DAS outputs are, if anything, slightly **more**
similar to each other than the raw closest-mic baselines.

This is strong evidence that plain DAS is not giving meaningful extra
separation in the known scene.

### Geometry-MVDR

MVDR improves average cross-beam differentiation:

- MVDR average pairwise log-magnitude correlation: `0.8150`

So MVDR is doing something real that DAS is not.

The directional response table also shows deeper nulls than DAS for some
off-axis directions, especially from the front/back beams toward the sides.

## What still looks weak

Even on the known example scene, separation is incomplete.

- `0deg_front` and `180deg_back` remain highly similar under MVDR
  (`0.9662` log-magnitude correlation)
- `90deg_right` and `270deg_left` are still fairly similar
  (`0.8944` log-magnitude correlation)

So MVDR helps, but it does **not** cleanly isolate the four talkers.

## Discovery-phase takeaway

The example-only validation supports the following conclusion:

- **Geometry-DAS is probably not worth further attention**
- **MVDR is better than DAS and better than the closest-mic baseline in a
  limited sense**
- **But even MVDR is still far from clean separation**

That means the beamforming path is still alive, but only narrowly. If it is
continued, it should be treated as:

- a validation exercise on the known scene first
- then possibly one last non-deep refinement

not as something already ready to transfer to `mixture.wav`

## Recommended listening order

For each direction:

1. `closest_<direction>.wav`
2. `geom_das_<direction>.wav`
3. `mvdr_<direction>.wav`

To play them in repo order, use:

`analysis/beam-forming/example-validation-01/listen_validation.py`

The main question is simple:

Does MVDR make the intended talker clearly more dominant than the closest mic,
or does it mostly add spatial coloration without enough real interference
suppression?
