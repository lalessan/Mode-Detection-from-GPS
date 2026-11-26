# Experiment Log

## Baseline (original student pipeline, 250m uniform stop radius)

| Metric | Value |
|--------|-------|
| ALL excl walk Jaccard | 0.7683 |
| ALL excl walk exact match | 70.7% |
| OOF macro F1 | 0.7775 |
| CAR→CAR (Level 4) | 69.9% |
| BUS→BUS (Level 4) | 71.2% |
| SUBWAY→SUBWAY (Level 4) | — |
| SUBWAY OOF F1 | 0.66 |

Source: validate_pipeline.py Level 3 + Level 4 output from original code.

---

## Experiment 1: Reduce stop radius 1000m → 50m (uniform)

**Change**: `stop_search_radius_m` from 1000 → 50 in validate_pipeline.py config.

**Result**: Ran, but SUBWAY features collapsed. DURATION_BUS BUS/CAR ratio improved (1.5x → 4.7x at 30m), but METRO features degraded because underground GPS noise means metro stops need wider radius.

**Verdict**: Partial improvement. Need per-mode radii.

---

## Experiment 5: Post-processing (CAR→BUS reclassification)

**Change**: postprocessing.py with 3 rules: CAR→BUS based on bus evidence, sandwich repair, segment merging.

**Result**: Rule 1 reclassified 28,683 segments CAR→BUS — far too aggressive. CAR→CAR dropped to 21.5%.

**Verdict**: FAILED. Disabled (`postprocess_enabled=False`).

---

## Run 2026-11-21a: Segment-level training + per-mode radii + bus seq (all combined)

**Changes**: Segment-level training with walking relabeling, per-mode stop radii, bus seq features.

**Result**: Everything worse. Macro F1 0.78→0.62. SUBWAY F1 0.66→0.23. 97.5% of single-modal trips falsely segmented.

**Verdict**: FAILED. Segment-level training is too noisy. Reverted.

---

## Run 2026-11-21b: Trip-level training + per-mode radii + bus seq

**Changes** (kept from previous):
- Per-mode stop radii (50m bus/tram/rail, 250m metro)
- Bus route sequence features (`BUS_SEQ_STOPS`, `BUS_SEQ_ROUTES`) at 30m
- Post-processing disabled

**Training**: Trip-level (original approach), 50,378 trips, 61 features, 5-fold CV.

**Level 3 (OOF):**

| Mode | Baseline F1 | This run F1 | Delta |
|------|-------------|-------------|-------|
| BIKE | 0.8199 | 0.8005 | -0.0193 |
| BUS | 0.6636 | **0.6841** | **+0.0205** |
| CAR | 0.9363 | 0.9320 | -0.0043 |
| SUBWAY | 0.6574 | 0.3670 | **-0.2904** |
| TRAIN | 0.8111 | **0.8252** | **+0.0141** |
| TRAMWAY | 0.6740 | **0.7464** | **+0.0724** |
| WALKING | 0.8799 | 0.8698 | -0.0101 |
| **MACRO** | **0.7775** | **0.7464** | **-0.0310** |

**Level 4 (multi-modal, the key metric):**

| Metric | Baseline | This run | Delta |
|--------|----------|----------|-------|
| ALL excl walk Jaccard | 0.7683 | **0.7740** | **+0.0057** |
| ALL excl walk exact | 70.7% | **71.2%** | **+0.5%** |
| SM excl walk exact | — | 74.3% | — |
| CAR→CAR | 69.9% | **83.2%** | **+13.3%** |
| BUS→BUS | 71.2% | 23.1% | -48.1% |
| TRAIN→TRAIN | — | 33.9% | — |
| SUBWAY→SUBWAY | — | 39.3% | — |
| TRAMWAY→TRAMWAY | — | 37.4% | — |

**Feature importance (top 5):** MIN_DIST_RAIL (113), tram_match_pct (67), DURATION_TRAM (57), SPEED_STD (40), SPEED_AVG (33). BUS_SEQ_STOPS ranked #16 (8.2).

**Key observations:**
- **Above baseline on ALL excl walk** (Jaccard 0.7740 > 0.7683). First time.
- CAR→CAR massively improved (83.2% vs 69.9%).
- BUS, TRAIN, TRAMWAY OOF F1 improved.
- **SUBWAY OOF F1 collapsed** (0.66→0.37). SUBWAY→WALKING confusion 38.1%. Per-mode 250m not enough.
- BUS→BUS Level 4 exact dropped (71.2%→23.1%) — BUS trips predicted as BUS+CAR (22.7%).

**Verdict**: Net positive on the main metric. SUBWAY is the main problem remaining.

---

## Radius test 2026-11-21: Optimal per-mode stop radii (test_stop_radii.py)

Tested radii 50, 100, 250, 500, 750, 1000m on 10% sample. Measured discriminability ratio (how well each feature separates its target mode from others).

| Stop type | Best radius | Avg ratio | Key insight |
|-----------|-------------|-----------|-------------|
| BUS (bus) | **250m** | 2.25 | MIN_DIST best at 250m (3.17), DURATION best at 50m (2.07) |
| SUBWAY (metro) | **750m** | 1.91 | MIN_DIST best at 750m (1.26), DURATION best at 100m (2.65). Flat 500-1000m. |
| TRAIN (rail) | **50m** | 24.96 | DURATION dominates (46.26 at 50m), MIN_DIST best at 1000m (7.36) |
| TRAMWAY (tram) | **50m** | 15.43 | DURATION dominates (26.71 at 50m), MIN_DIST best at 1000m (10.27) |

**Previous config**: bus=50m, metro=250m, rail=50m, tram=50m
**New config**: bus=250m, metro=750m, rail=50m, tram=50m

SUBWAY's poor discrimination (best ratio only 1.91) explains why it's hard to classify — metro stops just aren't very discriminative even at optimal radius.

---

## Run 2026-11-21c: Empirically optimized radii (bus=250m, metro=750m)

**Changes vs 21b**: bus 50m→250m, metro 250m→750m. Rail and tram stay at 50m.

**Level 3 OOF:**

| Mode | Baseline | Run 21b | This run (21c) | Delta vs 21b |
|------|----------|---------|----------------|--------------|
| BIKE | 0.8199 | 0.8005 | 0.8051 | +0.0046 |
| BUS | 0.6636 | 0.6841 | 0.6865 | +0.0024 |
| CAR | 0.9363 | 0.9320 | 0.9326 | +0.0006 |
| SUBWAY | 0.6574 | 0.3670 | 0.3756 | +0.0086 |
| TRAIN | 0.8111 | 0.8252 | 0.8262 | +0.0010 |
| TRAMWAY | 0.6740 | 0.7464 | 0.7437 | -0.0027 |
| WALKING | 0.8799 | 0.8698 | 0.8706 | +0.0008 |
| **MACRO** | **0.7775** | **0.7464** | **0.7486** | **+0.0022** |

**Level 4:**

| Metric | Baseline | Run 21b | This run (21c) | Delta vs 21b |
|--------|----------|---------|----------------|--------------|
| ALL excl walk Jaccard | 0.7683 | **0.7740** | 0.7716 | -0.0024 |
| ALL excl walk exact | 70.7% | **71.2%** | 70.9% | -0.3% |
| SUBWAY→SUBWAY | — | 39.3% | **46.0%** | +6.7% |
| CAR→CAR | 69.9% | 83.2% | **83.8%** | +0.6% |

**Feature importance:** STOPS_CLOSEST_METRO_P95 ranked #9 (18.9) — up from METRO_P70 at #10 (16.6). BUS_SEQ_STOPS up to #12 (11.2) from #16 (8.2).

**Verdict:** Marginal. OOF macro F1 slightly better (+0.002), SUBWAY improved (+6.7% Level 4), but overall Jaccard slightly worse (-0.002). Wider bus radius may add noise. The two runs are essentially equivalent — SUBWAY is fundamentally hard (best discriminability ratio only 1.91).

**Current best remains Run 21b** (Jaccard 0.7740, exact 71.2%).

---

## Run 2026-11-22a: Max-range filter + metro endpoint features

**Changes vs 21c (current best = 21b):**
1. **Stationary trip filter**: Replace displacement+distance filter with MAX_RANGE_M < 150m (non-walking only). Max range = furthest GPS point from trip centroid. From error analysis: catches 75.6% of CAR→WALKING errors, loses only 1.9% correct CAR. GPS drift inflates total distance but not max range.
2. **Metro endpoint features**: `METRO_DIST_START_M`, `METRO_DIST_END_M` — distance from first/last GPS point to nearest metro station. From error analysis: correct SUBWAY median 195m/139m, true WALKING median 1151m/1160m. SUBWAY→WALKING errors also near metro (205m/157m) — the model just doesn't have this signal yet.

**Hypothesis:**
- CAR→WALKING errors (n=1188, 5.2% of CAR) should drop substantially — these are mostly parked cars with GPS drift.
- SUBWAY→WALKING errors (n=750, 38.3% of SUBWAY) should improve — metro endpoint distance is highly discriminative (6x difference in medians).
- OOF macro F1 should improve from 0.7486 baseline.

**Result:**

Filter dropped 2,128 stationary trips (52,366 → 50,238 GPS trips, 48,388 after mode drop).

Level 3 OOF (48,388 trips, 63 features):

| Mode | Baseline | Run 21b | This run | Delta vs 21b |
|------|----------|---------|----------|--------------|
| BIKE | 0.8199 | 0.8005 | **0.8432** | +0.0427 |
| BUS | 0.6636 | 0.6841 | **0.7026** | +0.0185 |
| CAR | 0.9363 | 0.9320 | **0.9565** | +0.0245 |
| SUBWAY | 0.6574 | 0.3670 | 0.3990 | +0.0320 |
| TRAIN | 0.8111 | 0.8252 | **0.8522** | +0.0270 |
| TRAMWAY | 0.6740 | 0.7464 | 0.7562 | +0.0098 |
| WALKING | 0.8799 | 0.8698 | **0.9232** | +0.0534 |
| **MACRO** | **0.7775** | **0.7464** | **0.7761** | **+0.0297** |

Level 4 (multi-modal):

| Metric | Baseline | Run 21b | This run | Delta vs 21b |
|--------|----------|---------|----------|--------------|
| ALL excl walk Jaccard | 0.7683 | 0.7740 | **0.8227** | **+0.0487** |
| ALL excl walk exact | 70.7% | 71.2% | **76.7%** | **+5.5%** |
| SM excl walk exact | — | 74.3% | **80.1%** | **+5.8%** |
| CAR→CAR | 69.9% | 83.2% | **87.9%** | **+4.7%** |
| BUS→BUS | 71.2% | 23.1% | 27.1% | +4.0% |
| SUBWAY→SUBWAY | — | 39.3% | 35.6% | -3.7% |
| TRAIN→TRAIN | — | 33.9% | 29.7% | -4.2% |
| TRAMWAY→TRAMWAY | — | 37.4% | 31.7% | -5.7% |

Feature importance: METRO_DIST_END_M #13 (10.9), METRO_DIST_START_M #17 (6.4).

**Key observations:**
- **Massive Level 4 improvement**: ALL excl walk Jaccard 0.7740 → 0.8227 (+0.049), exact 71.2% → 76.7% (+5.5%). Best result by far.
- **CAR→CAR at 87.9%** — up from 83.2%. Max-range filter removed noisy stationary trips from training.
- **Every mode improved at OOF level** vs Run 21b. Macro F1 recovered to near-baseline (0.7761 vs 0.7775).
- **SUBWAY still weak**: OOF F1 only 0.40, Level 4 exact 35.6% (down from 39.3%). Metro endpoint features ranked #13/#17 — model uses them but SUBWAY→WALKING confusion persists (37% at Level 4).
- **Caveat**: Part of the OOF improvement is from removing 2,128 hard-to-classify trips from evaluation, not purely from better classification.

**Verdict**: New best on the main metric. **CURRENT BEST.**

---

## Confidence analysis (Run 22a)

Added prediction confidence (max predicted probability) to both trip-level OOF and segment-level multi-modal predictions.

**Trip-level OOF (Level 3, 48,388 trips):**

| Threshold | Kept | Coverage | Accuracy | Macro F1 |
|-----------|------|----------|----------|----------|
| 0.00 | 48,388 | 100% | 89.8% | 0.7761 |
| 0.50 | 47,402 | 98.0% | 90.7% | 0.7893 |
| 0.70 | 43,400 | 89.7% | 93.3% | 0.8217 |
| 0.80 | 40,578 | 83.9% | 94.7% | 0.8443 |
| 0.90 | 35,784 | 74.0% | 96.2% | 0.8695 |
| 0.95 | 30,837 | 63.7% | 97.3% | 0.8871 |

Per-mode confidence gap (median correct - median incorrect): WALKING +0.333, BIKE +0.253, CAR +0.195, TRAMWAY +0.182, BUS +0.145, TRAIN +0.136, **SUBWAY -0.002** (model equally confident whether right or wrong).

**Segment-level multi-modal (Level 4, 50,095 trips, min-confidence = lowest segment confidence):**

| Threshold | Kept | Coverage | Jaccard nw | Exact nw |
|-----------|------|----------|------------|----------|
| 0.00 | 50,095 | 100% | 0.8197 | 76.7% |
| 0.50 | 46,553 | 92.9% | 0.8435 | 80.2% |
| 0.70 | 37,297 | 74.5% | 0.9032 | 88.5% |
| 0.80 | 32,906 | 65.7% | 0.9254 | 91.5% |
| 0.90 | 26,739 | 53.4% | 0.9510 | 94.6% |
| 0.95 | 21,164 | 42.2% | 0.9668 | 96.5% |

**Key finding**: Confidence filtering is highly effective. At 0.70 min-confidence, 74.5% of trips are kept with Jaccard excl walking 0.90 (vs 0.82 unfiltered) and exact match 88.5% (vs 76.7%). SUBWAY is the only mode where confidence does not discriminate correct from incorrect predictions.

---

---
---

# Google Timeline Validation (Copenhagen)

Validation of the Paris-trained model (Run 22a) on personal Google Timeline data (~1.2M records, 2014–2020, Copenhagen). Tests real-world transferability: different city, different data source, no ground truth labels — only Google's activity recognition for comparison.

---

## Setup

### Pipeline

1. Parse raw GPS from Google Timeline JSON (320K records in Copenhagen bbox)
2. **infostop** stop detection (r1=30m, min_staying_time=300s) → 677 stops, 1500 trips
3. Filters aligned with training: min_points=10, min_duration=180s, gap=300s, max_range=150m (dropped 152 stationary trips)
4. **Trip quality filter**: max_jump=2km, max_speed=500km/h, jump-back detection, sparse sampling (median interval >120s). Removed 287/1500 trips (kept 1213)
5. `model.predict_proba(gps, segment=True)` → segments with iterative speed filter
6. Per-segment Google label assignment via `segmented_gps()` (direct GPS→segment mapping)

### Evaluation methodology

- **Flexible matching**: IN_VEHICLE accepts CAR or BUS; IN_RAIL_VEHICLE accepts TRAIN, SUBWAY, TRAMWAY
- **Excluded from eval**: IN_TWO_WHEELER_VEHICLE, RUNNING
- **Verifiable only**: excludes trips where we predict SUBWAY/TRAIN/TRAMWAY (Google has no rail labels)

### Infostop vs Google STILL validation

| Metric | Value |
|--------|-------|
| Precision | 67.2% (of infostop stops, 67% are Google STILL) |
| Recall | 91.3% (of Google STILL, 91% are infostop stops) |
| F1 | 0.774 |

False positives mostly TILTING(13.5K) and UNKNOWN(6.1K) — ambiguous activities, not real movement. High recall means infostop catches nearly all stationary periods.

### Google label limitations

Google only provides 3 transport modes: WALKING (72%), BIKE (17%), CAR (11%). No public transport labels (SUBWAY, TRAIN, BUS). Google confidence is high: median 95, 69% above 90.

---

## GT-Run 1–3: Initial experiments (old model, before quality filter)

These runs used the original model (47 features, no endpoint constraints) and no trip quality filter.

| Run | Features | Overall | Verifiable | SUBWAY pred. | Notes |
|-----|----------|---------|------------|-------------|-------|
| GT-1 | Motion only | 36.3% | 64.2% | 409 | Baseline |
| GT-2 | OSM routes only | 31.8% | 60.5% | 450 | OSM hurt — dense transit in Copenhagen |
| GT-3 | Full (GTFS+OSM) | 36.0% | 52.9% | 160 | GTFS fixed SUBWAY, OSM added noise |

---

## GT-Run 4: Quality filter + old model (full features)

Added trip quality filter (max_jump=2km, max_speed=500km/h, jump-back, sparse). 1500 → 1213 trips.

| Metric | Value |
|--------|-------|
| Overall agreement | 42.6% |
| Verifiable only | 56.7% |
| Verifiable + conf ≥ 0.7 | 62.1% |
| SUBWAY predictions | 145 |
| TRAIN predictions | 154 |

Quality filter removed the worst GPS noise but didn't improve verifiable agreement much — the main bottleneck is SUBWAY/TRAIN over-prediction.

---

## GT-Run 5: Retrained model with endpoint features (full features)

Retrained on Paris with new features:
- `{METRO,TRAIN,TRAM}_DIST_START_M`, `_DIST_END_M`, `_DIST_EP_MAX_M`
- EP_MAX = max(start, end) distance — both endpoints must be near a station
- 70 features total (up from 63)

Training impact: `METRO_DIST_EP_MAX_M` → rank #11 importance (gain 15.7). SUBWAY OOF F1 dropped 0.66→0.41 (model more conservative). All other modes improved.

| Metric | GT-Run 4 (old) | GT-Run 5 (new) |
|--------|----------------|----------------|
| Overall agreement | 42.6% | **44.9%** |
| Verifiable only | 56.7% | **59.2%** |
| Verifiable + conf ≥ 0.7 | 62.1% | **64.8%** |
| Verifiable + conf ≥ 0.8 | 60.9% | **69.5%** |
| SUBWAY predictions | 145 | **82** (−43%) |
| TRAIN predictions | 154 | 206 (+34%) |
| IN_VEHICLE match (verif.) | 56.9% | **60.6%** |
| ON_FOOT match (verif.) | 59.3% | **61.8%** |
| ON_BICYCLE match (verif.) | 46.4% | **47.7%** |

---

## GT-Run 6: Station endpoint constraint (post-processing Rule 0)

Added hard post-processing constraint: SUBWAY/TRAIN/TRAMWAY predictions are rejected when endpoints are too far from stations. Applied at the probability level — zeroes out the constrained class and renormalizes, so `idxmax()` picks the next-best mode.

Thresholds: SUBWAY 500m, TRAIN 1000m, TRAMWAY 300m (from nearest station to both start and end points via `{MODE}_DIST_EP_MAX_M`).

Affected 2903 segment×mode combinations (many segments constrained for multiple modes).

| Metric | GT-Run 5 | GT-Run 6 | Delta |
|--------|----------|----------|-------|
| Overall agreement | 44.9% | **47.5%** | **+2.6pp** |
| Verifiable only | 59.2% | 54.8% | −4.4pp |
| Verifiable coverage | 893 (76%) | **1021 (87%)** | **+11pp** |
| Verifiable + conf ≥ 0.7 | 64.8% | 57.0% | −7.8pp |
| Verifiable + conf ≥ 0.8 | 69.5% | 56.5% | −13.0pp |
| SUBWAY predictions | 82 | **53** (−35%) |  |
| TRAIN predictions | 206 | **110** (−47%) |  |
| BUS predictions | 20 | **38** (+90%) |  |
| CAR predictions | 261 | **356** (+36%) |  |
| IN_VEHICLE match | 42.1% | **64.2%** | **+22pp** |
| ON_FOOT match | 47.7% | 48.6% | +0.9pp |
| ON_BICYCLE match | 34.9% | 35.4% | +0.5pp |

**Key observations:**
- **IN_VEHICLE matching jumped 42→64%**: Many false TRAIN predictions correctly reassigned to CAR.
- **SUBWAY cut by 35%, TRAIN by 47%**: Endpoint constraint is effective at removing physically impossible predictions.
- **BUS doubled (20→38)**: Ex-SUBWAY/TRAIN segments now assigned to BUS where bus features support it.
- **Verifiable agreement dropped (59.2→54.8%)**: More trips are now verifiable (1021 vs 893), but the new CAR predictions include many wrong ones (ON_BICYCLE→CAR, ON_FOOT→CAR).
- **High-confidence verifiable dropped sharply (69.5→56.5%)**: Renormalized confidences after zeroing rail modes are inflated — a segment may show 0.95 for CAR just because TRAIN was zeroed, not because it's truly confident.

**Verdict:** Overall agreement improved (+2.6pp), and the constraint makes physically sensible predictions. But confidence scores are no longer well-calibrated after renormalization — high-confidence filtering is less effective. The ON_BICYCLE→CAR confusion (72/212 trips) is now the biggest problem for verifiable trips.

---

## Key findings

1. **GPS quality filter is essential**: Google Timeline has cell tower bouncing (jumps of 6–13km), sparse sampling (~60s), and teleportation artefacts. Removed 287/1500 trips (19%).
2. **Iterative speed filter**: Single-pass speed filtering misses cascading jumps. Iterating (remove outliers → recompute speed → repeat) catches more.
3. **Endpoint distance features work**: `METRO_DIST_EP_MAX_M` became rank #11 feature, reduced SUBWAY predictions by 43% (GT-Run 5).
4. **Hard endpoint constraint highly effective**: Post-processing Rule 0 (GT-Run 6) cut SUBWAY by 35% and TRAIN by 47% further. IN_VEHICLE match jumped 42→64%.
5. **Google labels are coarse**: Only 3 types (ON_FOOT 74%, ON_BICYCLE 18%, IN_VEHICLE 8%). No rail or bus labels. This limits what we can validate.
6. **Google confidence is high** (median 96) — disagreements aren't from Google uncertainty.
7. **ON_BICYCLE weakest** (35.4% match): 72/212 bike trips predicted as CAR. Likely a Paris→Copenhagen transferability gap. Worsened after endpoint constraint pushed ex-TRAIN segments to CAR.
8. **Confidence calibration breaks after proba renormalization**: After zeroing constrained modes, renormalized confidences are inflated. High-confidence filtering lost effectiveness (69.5→56.5% at ≥0.8).

---

## Open issues

1. **ON_BICYCLE → CAR confusion**: 72/212 bike trips predicted CAR (34%). Biggest source of error in verifiable trips. Paris→Copenhagen transferability gap (cycling infrastructure, speeds).
2. **ON_FOOT → CAR misclassification**: 219/870 ON_FOOT trips predicted as CAR (25%). Worsened after endpoint constraint redirected ex-TRAIN to CAR.
3. **Confidence calibration after renormalization**: Zeroing constrained modes inflates remaining probabilities. Need original (pre-renorm) confidence for filtering, or recalibrate.
4. **BUS Level 4 (Paris training)**: Trip-level exact dropped from 71.2% to 23.1%. Short segments lose the discriminative stopping pattern.
5. **SUBWAY OOF F1 dropped to 0.41**: Model conservative about SUBWAY — good for reducing false positives but hurts recall on real subway trips.
6. **Note on Level 3 comparison**: Original uses 80/20 split (10,637 test trips), modedetect uses OOF on 48,388 trips. OOF is a harder evaluation.
