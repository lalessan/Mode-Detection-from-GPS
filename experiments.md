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

## Run 2026-11-21b: Trip-level training + per-mode radii + bus seq (**CURRENT BEST**)

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

## Open issues

1. **BUS Level 4**: Trip-level exact dropped from 71.2% to 23.1%. Segments within BUS trips get classified as CAR (BUS+CAR = 22.7%). The classifier knows BUS at trip level (OOF F1 0.68) but short segments lose the discriminative pattern.

2. **Note on Level 3 comparison**: Original uses 80/20 split (10,637 test trips), modedetect uses OOF on 50,378 trips. OOF is a harder evaluation — every trip is tested on a fold that didn't see it. The comparison is not perfectly apples-to-apples.
