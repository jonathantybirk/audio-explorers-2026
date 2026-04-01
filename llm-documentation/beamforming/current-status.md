# Current Beamforming Status

This note is about the current analysis code, not the case itself.

## Short Answer

The current code now matches the case on the azimuth convention:

- `0° = front`
- `90° = left`
- `180° = back`
- `270° = right`

That part is high confidence.

The current code does **not** fully stay within what the case guarantees.
Several parts of the beamforming pipeline still depend on inferred geometry and modeling assumptions.

## What I Am Confident About

These parts now match the case:

- channel order is treated as `[LF, LR, RF, RR]`
- the microphone labels match Figure 1
- `example_mixture.wav` is treated as the labeled reference scene
- `mixture.wav` is treated as the unlabeled inference scene
- the steering labels now use `90deg_left` and `270deg_right`

Relevant code:

- [geom-das-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/geom-das-01/geom-das-01.py#L20)
- [mvdr-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/mvdr-01/mvdr-01.py#L41)
- [example-validation-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/example-validation-01/example-validation-01.py#L40)

## What Still Does Not Come From The Case

The case does **not** provide exact microphone spacing, but the geometry-based code assumes:

- `TAU_LR = 29` samples
- `TAU_FR = 8` samples

Those values are inferred from the recordings, not given by the case.

The case also does **not** guarantee:

- a far-field propagation model
- a rectangular 2D mic layout in sample units
- that `mixture.wav` should be analyzed only at four cardinal steering directions
- that `mixture.wav` contains exactly four speakers

But the current code assumes some or all of those things.

Relevant code:

- [geom-das-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/geom-das-01/geom-das-01.py#L47)
- [mvdr-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/mvdr-01/mvdr-01.py#L31)
- [mvdr-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/mvdr-01/mvdr-01.py#L14)

## Script-By-Script Status

### `gcc-das-01`

This is the least assumption-heavy script.

What it does well:

- calibrates delays directly from `example_mixture.wav`
- does not require explicit physical mic spacing
- now uses the corrected left-positive azimuth convention

What is still weak:

- it still transfers the four labeled example directions directly into beam labels for `mixture.wav`
- the measured GCC peaks are not cleanly symmetric
- in practice, the recovered left/right delays are uneven, which means the direction labels are still fragile

Current interpretation:

- useful as an exploratory empirical baseline
- not strong enough to treat as a reliable DoA solution

Relevant code:

- [gcc-das-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/gcc-das-01/gcc-das-01.py#L91)

### `geom-das-01`

This script is assumption-heavy.

What it assumes:

- an explicit mic geometry reconstructed from inferred delays
- a far-field analytic steering model
- four fixed steering directions only

Current interpretation:

- internally consistent with the corrected azimuth convention
- not purely case-backed, because the geometry is inferred rather than provided

Relevant code:

- [geom-das-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/geom-das-01/geom-das-01.py#L54)
- [geom-das-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/geom-das-01/geom-das-01.py#L67)

### `mvdr-01`

This is the most assumption-heavy of the current beamforming scripts.

What it assumes:

- the same inferred geometry as `geom-das-01`
- the same far-field steering model
- covariance estimated blindly from the full unlabeled mixture
- four fixed steering directions

Current interpretation:

- mathematically consistent with the corrected azimuth convention
- still not something I would describe as fully case-backed

Relevant code:

- [mvdr-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/mvdr-01/mvdr-01.py#L60)
- [mvdr-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/mvdr-01/mvdr-01.py#L101)

### `example-validation-01`

This is the best-structured part of the current analysis.

What it does well:

- validates only on the labeled example scene
- keeps `mixture.wav` out of the validation loop
- compares closest-mic, geometry-DAS, and MVDR under the corrected angle convention

What is still limited:

- the closest-mic mapping for cardinal directions is a manual tie-break
- the validation still inherits the geometry assumptions from `geom-das-01` and `mvdr-01`

Relevant code:

- [example-validation-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/example-validation-01/example-validation-01.py#L47)
- [example-validation-01.py](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/example-validation-01/example-validation-01.py#L235)

## What The Validation Suggests

The current validation does **not** show a clean, convincing beamforming win.

From [analysis/beam-forming/example-validation-01/summary.txt](/Users/jonathantybirk/Desktop/Audio%20Explorers%202026/analysis/beam-forming/example-validation-01/summary.txt):

- MVDR reduces average pairwise correlation more than the baselines, which means it produces more differentiated outputs.
- But the directional response tables still show substantial front/back leakage and uneven left/right behavior.

That means:

- the code is now labeled more correctly
- but the underlying beamforming result is still only exploratory

## Bottom Line

I am sure about this:

- the old `90° = right` interpretation was wrong
- the current left-positive azimuth convention now matches the case figure

I am **not** sure about this:

- that the current geometry-based beamforming pipeline is the right physical model
- that the current beam labels on `mixture.wav` correspond to true source directions with high confidence
- that the current analysis should be presented as if it follows only from the case

The safe way to describe the current state is:

- `setup.md` = what the case guarantees
- current beamforming code = exploratory modeling built on top of those facts
