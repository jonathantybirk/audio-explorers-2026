# Report Outline

## 1. Introduction (~0.3p)
- Hearing aid cocktail party problem
- Goal: identify 7 speakers, estimate DoAs, identify talker of interest, produce enhanced output

---

## 2. Validation on Example Mixture (~0.8p)
- Microphone array geometry: 4-channel hearing aid, inter-ear baseline, intra-ear spacing, channel order
  - Why front/rear is fundamentally unresolvable: <16mm baseline, <2-sample TDOA difference
  - Figure 1: array diagram from case
- Interactive comparison tool built for systematic method evaluation
  - Figure 2: demo screenshot
- Beamforming: natural first attempt. Failed — array too small, insufficient spatial selectivity
- SRP-PHAT: blind DoA scanning. Validated against 4 known positions in example mixture
- FastMNMF2 + Bayesian optimisation (Optuna): clean 4-speaker separation confirmed
  - Front/rear labels swapped — consistent with geometry limitation above

---

## 3. Mixture Analysis (~1.2p)
- 7 speakers identified from BSS sweeps, named for reference throughout
- Rough transcriptions produced to aid identification
- First DoA observation: listening to individual L/R mic channels directly
  - Australia Man clearly left-dominant, Brunch Woman right-dominant
- Inter-ear spectro-temporal binary masking:
  - Per TF-bin left/right power ratio; bins dominated by one ear killed (>15% imbalance)
  - 15.7% of bins preserved, applied to all 4 channels
- FastMNMF2 on masked 4-channel signal → remaining 5 speakers separated more cleanly
- Activity-weighted SRP-PHAT DoA estimation:
  - Each isolated BSS source used as frame-activity mask
  - SRP-PHAT accumulated on 4-channel signal during active frames only
  - Second-peak analysis for confidence
- Talker of interest identification:
  - Convo Woman markedly louder than all others → closest source → most likely the hearing aid wearer
  - Convo Man is her direct conversational partner → talker of interest

---

## 4. Target Enhancement (~0.5p)
- FastMNMF2 n=4 on inter-ear masked signal → Convo Man isolated in source 3
- Bayesian optimisation (Optuna, 60 trials, HPC) sweeping:
  - STFT size, hop size, number of iterations, NMF components
  - Objective: minimise pairwise cross-correlation sum across sources
- Spectral noise reduction post-processing (noisereduce, non-stationary, prop=0.8)

---

## 5. Results (~0.5p)

| Speaker          | Gender | Est. DoA    | Notes                  |
|------------------|--------|-------------|------------------------|
| Convo Man        | M      | ~0° front   | **Talker of interest** |
| Convo Woman      | F      | ~0° front   | Likely wearer          |
| Australia Man    | M      | ~90° left   | Confirmed              |
| Brunch Woman     | F      | ~270° right | Confirmed              |
| Mountain Man     | M      | ~0° front   | Strong                 |
| Ageing Man       | M      | ~180° back  | Moderate               |
| Burning House Man| M      | ~180° back  | Estimated              |

- Microphone selection: BSS output combines all 4 channels via spatial covariance — preferred over any single mic for a near-frontal source
- Delivered: target_speaker_enhanced.wav

---

## 6. Going Further (~0.2p)
- Fine-tuning SepFormer/TF-GridNet on synthetic 7-speaker Libri7Mix dataset (HPC, A100 GPU)
- Data generation pipeline completed
- Storage quota approval came too late for training to complete before deadline

---

## Appendix A
Full transcripts of all 7 speakers










Now we have proper results in the correct convention:

Speaker	DoA #1	DoA #2	Confidence
Australia man	97.5° (left)	82.5°	✓ confirmed
Brunch woman	258.5° (right)	277.5°	✓ confirmed
Mountain man	1.0° (front)	0.0°	strong — both peaks agree
Convo man (s3)	181.0° (back)	0.0°	moderate
Ageing man	180.5° (back)	97.5°	moderate
Burning house	258.5°	284.0°	weak — source contaminated by BW residual


