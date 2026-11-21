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

## Open issues

1. **SUBWAY**: OOF F1 0.37, Level 4 exact 39.3%. 38% misclassified as WALKING. Underground GPS noise makes stop proximity unreliable. May need:
   - Even wider metro radius (500m?)
   - subway_match_pct is only ranked #26 — route matching may also be weak for metro
   - Dedicated subway features (e.g. speed pattern: fast underground, stops at stations)

2. **BUS Level 4**: Trip-level exact dropped from 71.2% to 23.1%. Segments within BUS trips get classified as CAR (BUS+CAR = 22.7%). The classifier knows BUS at trip level (OOF F1 0.68) but short segments lose the discriminative pattern.

3. **Note on Level 3 comparison**: Original uses 80/20 split (10,637 test trips), modedetect uses OOF on 50,378 trips. OOF is a harder evaluation — every trip is tested on a fold that didn't see it. The comparison is not perfectly apples-to-apples.
