# Speaker Profiles — mixture.wav

## Shorthands

| Shorthand | Gender | Language | Transcript excerpt |
|-----------|--------|----------|--------------------|
| **mountain man** | M | English | "I had never thought about climbing mountains" |
| **australia man** | M | English | "Most sightseers will be Chinese, and for the great majority, this 83 million monument to life down under, was the closest they'd come to visit Australia" |
| **ageing man** | M | English | "It is inevitable, the muscles weaken, hearing and vision fade, we get wrinkled and [stooped?], we can't [eat?] run or even walk as fast as we used to" |
| **brunch woman** | F | English | "Every year at my birthday brunch, guests elbow each other over the smoked salmon" |
| **convo man** | M | English | "Hi, how are you? ... I just had my birthday. ... I'm 35" |
| **convo woman** | F | English | "Great, how are you? ... Really, how old are you?" |
| **burning house man** | M | English | "What would you say, if your house was burning down? If a fire hit my house, I would be running around, picking up the things" |

Notes on transcripts:
- Ordering above is arbitrary — does NOT correspond to DoA or source index
- Transcripts are approximate; [word?] = uncertain
- All 7 speakers are unique individuals
- All clips continue beyond what is written above
- The conversation (convo man/convo woman) is a strict back-and-forth exchange — they NEVER overlap, they alternate turns

---

## DoA estimates (SRP-PHAT on raw mixture)

Convention: 0° = front, 90° = left, 180° = back, 270° = right, CCW

| SRP peak | Azimuth | Rel. power | Confidence | Notes |
|----------|---------|------------|------------|-------|
| S1 | 2° | 1.00 | Low | Near-front; front/back ambiguous (1.6cm intra-ear baseline) |
| S2 | 25° | 0.40 | Low | Slightly left of front; possibly mirror at ~155° |
| S3 | 90° | 0.75 | **High** | Left — large inter-ear TDOA, unambiguous |
| S4 | 178° | 0.62 | **High** | Back — intra-ear sign is clearly negative |
| S5 | 270° | 0.75 | **High** | Right — large inter-ear TDOA, unambiguous |
| S6 | 332.5° | 0.40 | Low | Slightly right of front; possibly mirror at ~207° |
| S7 | 358° | 0.81 | Low | Near-front; front/back ambiguous |

**Time-windowed SRP-PHAT insight**: 90°, 180°, 270° dominate almost every 1s window throughout the 21s clip — these 3 sources are the most continuously active / energetic. Near-0° peaks (2°, 356°, 358°) appear intermittently, particularly around:
- t ≈ 1–5s (early cluster)
- t ≈ 6.5–7.5s
- t ≈ 14–15s
- t ≈ 16.5–18s (strongest near-front cluster)

This suggests the front speakers talk in shorter bursts (consistent with conversation turn-taking) while the background speakers are more continuous.

---

## Speaker → DoA mapping (work in progress)

| Speaker | Likely DoA | Evidence | Confidence |
|---------|------------|----------|------------|
| brunch woman | **~270° (right)** | GCC cross-corr LF/RF = 0.379 (2.6× stronger on RF). Raw mic listening confirms. fmnmf2-tuned DoA label was wrong (BSS permutation problem). | **High** |
| australia man | **~90° (left)** | GCC cross-corr LF/RF = 1.212 (stronger on LF/LR). Dominant in fmnmf2-opt source 2. | **High** |
| ageing man | **~0° or ~180° (front/back axis)** | GCC cross-corr LF/RF = 0.914 — nearly centered. fmnmf2-tuned label "178°" unreliable. Front vs back uncertain. | Low-Medium |
| mountain man | **~0° or ~180° (front/back axis)** | GCC cross-corr LF/RF = 0.892 — nearly centered (source 3 also contains burning house man). | Low |
| burning house man | **~0° or ~180° (front/back axis)** | Same as mountain man — co-dominant in source 3, LF/RF nearly centered. | Low |
| convo man | **~0° or ~180° (front/back axis)** | GCC cross-corr LF/RF = 0.877 — nearly centered. Front vs back uncertain. | Low |
| convo woman | **~0° or ~180° (front/back axis)** | Same as convo man — grouped in source 5, near front/back axis. | Low |

*Update this table as transcription timing becomes available.*

---

## Algorithm outputs — speaker presence notes

### FastMNMF2 Optuna n=7 ★ (best overall)
`mixture_fmnmf2_opt_n7_source_{1..7}.wav`

| Source | Clearly audible | Faint / underneath |
|--------|----------------|-------------------|
| 1 | **ageing man** (dominant) | convo woman (audible), convo man (harder) |
| 2 | **australia man** (dominant) | Unknown faint male |
| 3 | **mountain man** (clear), **burning house man** (clear) | — |
| 4 | **convo man** (clear), **brunch woman** (clear) | convo woman (fainter) |
| 5 | Noise | — |
| 6 | Noise (kinda CM/CW conversation?) | — |
| 7 | Noise | — |

→ Effectively 4 meaningful sources for 7 speakers. Sources 3 and 4 each blend 2–3 speakers.

### FastMNMF2 Optuna n=5
`mixture_fmnmf2_opt_n5_source_{1..5}.wav`

| Source | Clearly audible | Faint / underneath |
|--------|----------------|-------------------|
| 1 | **ageing man** (dominant) | convo man (clear), convo woman (especially clear) |
| 2 | **australia man** (dominant) | Other males underneath, hard to hear |
| 3 | **mountain man** (dominant), **burning house man** (quite clear) | — |
| 4 | **brunch woman** (dominant), **convo man** (clear) | convo woman (fainter) |
| 5 | **convo woman** (clearest), **convo man** (distinct) | Underlying mesh of voices |

→ Source 5 is the best isolation of the convo couple so far. Still not clean but clearly their dominant source.

### FastMNMF2 tuned n=5
`mixture_fmnmf2_tuned_source_{1..5}.wav`
*Not yet listened to.*

### ICA variants (mixture_ica_*, mixture_ica_tuned_*, mixture_ica_mtuned_*, mixture_ica_wiener_*)
*Not yet listened to for mixture. DoA-sorted into 4 buckets: ~134°, ~178°, ~313°, ~356°.*

### AuxIVA Optuna
`mixture_auxiva_opt_source_*.wav`
*Not yet listened to.*

---

## Experiments log

What we tried, what worked, what didn't. Chronological, factual.

### 1. SRP-PHAT on raw 4-channel mixture
`analysis/doa/estimate_doa_mixture.py`
Found 7 peaks: 2°, 25°, 90°, 178°, 270°, 332.5°, 358°. Verified correct on example_mixture (ground truth 0°/90°/180°/270° all recovered exactly). The 90° and 270° peaks are reliable (large inter-ear TDOA, no mirror ambiguity). All near-0° and near-180° peaks are uncertain because the 1.6cm intra-ear baseline barely resolves front vs back.

### 2. Masked SRP-PHAT on fmnmf2-opt-n7 sources
`analysis/doa/masked_srp_doa.py`
Used Wiener TF masks from the 7 separated mono sources to weight the raw-channel SRP-PHAT. Result: almost all sources blurred to ~132° (between 90° and 178°), regardless of source. Failed — the masks were too leaky because sources 1–4 each contain multiple speakers, so the masks overlapped and spatial information averaged out.

### 3. Time-windowed SRP-PHAT on raw mixture
`analysis/doa/windowed_srp_doa.py`
Divided the 21s clip into 1s windows (0.25s hop), ran SRP-PHAT per window. Found that 90°, 180°, 270° dominate almost every window throughout — those speakers talk continuously. Near-0° peaks appear intermittently (t≈1–5s, 6.5–7.5s, 14–15s, 16.5–18s), consistent with shorter conversational turns. Produced a heatmap (`windowed_doa_heatmap.png`). Useful as a lookup table but requires knowing *when* each speaker talks.

### 4. Listening to fmnmf2-tuned DoA-labeled sources
Sources are labeled by angle (e.g. `_90deg`, `_178deg`) — these labels came from SRP-PHAT on the BSS multichannel source images.

Tried to map: "who is clearest in the 90° source" → that speaker is at 90°.

**Finding: the DoA labels are unreliable.** Two reasons confirmed empirically:
- Brunch woman was labeled "90° (left)" but raw mic listening clearly shows her louder on RF/RR (right). GCC cross-correlation confirms she is right.
- In example_mixture (known ground truth), the English speaker (Nightingale woman) who is at 0° (front) per the case PDF is labeled as being behind by fmnmf2 tuned. Directly contradicts ground truth.

**Root cause:** Frequency-domain BSS methods (FastMNMF2, ICA) suffer from the permutation ambiguity — different frequency bins of the same true source can end up in different output channels in the multichannel source images. SRP-PHAT on those corrupted images gives unreliable DoAs. The mono outputs (averaged across channel images) are reliable as audio, but the spatial content of the images is not.

### 5. GCC cross-correlation: mono sources vs raw channels
Quick-and-dirty: for each fmnmf2-opt-n5 mono source, compute GCC-PHAT peak amplitude against each of the 4 raw mixture channels. The channel with highest coherence is the mic the source arrived at most directly → direction. Avoids BSS permutation problem because it only uses the mono source signal (reliable) cross-correlated with the untouched raw channels.

Results (LF/RF coherence ratio):
| Source | LF/RF ratio | Direction |
|--------|-------------|-----------|
| 1 (ageing man dom.) | 0.914 | nearly centered |
| 2 (australia man dom.) | 1.212 | LEFT |
| 3 (mountain+burning house men) | 0.892 | nearly centered |
| 4 (brunch woman dom.) | 0.379 | strongly RIGHT |
| 5 (convo couple) | 0.877 | nearly centered |

**Confirmed:** australia man = left (~90°), brunch woman = right (~270°). Consistent with raw mic listening (brunch woman faint but audible on LF/LR, clear on RF/RR; australia man very faint on RF/RR, clear on LF/LR). Ageing man, mountain man, burning house man, convo couple all near front/back axis — LF/RF nearly equal.

**Caveat:** the 2.6× coherence gap for brunch woman is compelling, but BSS contamination in the mono source (other right-side speakers bleeding in) could in principle inflate RF coherence. Assessed as probably right (medium-high confidence), not certain.

### 6. Channel RMS check
LF: 0.0704, LR: 0.0587, RF: 0.0601, RR: 0.0464. Left channels ~17% louder overall. Right channels are genuinely quieter — this means a left-side source being harder to hear on the right is expected, but cannot explain a 2.6× GCC coherence gap.

---

## Talker of interest hypothesis

The most likely scenario: the **hearing aid wearer is one of the conversation pair (CM or CW)**. The other conversant (the one in front, speaking *towards* the wearer's mics) would be louder in the recording and is the primary **talker of interest**.

- S1 (2°, power 1.0) — the most energetic near-front source. Could be the conversant directly in front (convo man or convo woman).
- S7 (358°, power 0.81) — the second near-front source. Could be the wearer's own voice.
- The stronger-in-recording speaker of the two would be the one facing the mics (the target).

This remains a hypothesis until CM/CW DoAs are confirmed via transcription timing + windowed SRP.
