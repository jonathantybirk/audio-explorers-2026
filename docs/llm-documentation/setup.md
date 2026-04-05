# Setup

This note only states facts that are directly supported by the case PDF.

## Recordings

- `mixture.wav` is the main task recording.
- `example_mixture.wav` is a separate example recording from the same microphone array.
- Both recordings use the same 4-channel channel order:
  `[left front, left rear, right front, right rear]`.

## Microphone Positions

From Figure 1, the microphone layout is:

- `LF`: left front
- `LR`: left rear
- `RF`: right front
- `RR`: right rear

What the case guarantees:

- There are two microphones on the left ear and two microphones on the right ear.
- Each ear has a front microphone and a rear microphone.

What the case does not give:

- No exact microphone spacing in meters.
- No exact head dimensions.
- No microphone directivity model.

The safe interpretation is that the case specifies microphone positions on the head, not a separate set of microphone look angles.

## Azimuth Convention

Figure 2 shows the azimuth angle `φ` measured from the front axis around the listener.

The consistent reading of the figure is:

- `0°` = front
- `90°` = left
- `180°` = back
- `270°` = right

## Known Talker Positions In `example_mixture.wav`

The case explicitly states that the example scene contains four talkers:

- one at `0°`
- one at `90°`
- one at `180°`
- one at `270°`

The case also states:

- all four are approximately `1.9 m` from the listener
- three talkers are Danish-speaking
- one talker is English-speaking

From Figure 2, the language placement appears to be:

- `0° front`: English-speaking talker
- `90° left`: Danish-speaking talker
- `180° back`: Danish-speaking talker
- `270° right`: Danish-speaking talker

## What Is Known About `mixture.wav`

From the case, we know:

- it is recorded with the same microphone array
- it is the main unlabeled multi-talker scene

From the case, we do not know:

- the number of talkers
- the talker azimuths
- the talker distances
- the talker languages
- the talker genders
- which talker is the talker of interest

## Practical Consequence

`example_mixture.wav` is the calibration/reference scene because its talker positions are given.
`mixture.wav` is the inference scene because its talker positions are not given.
