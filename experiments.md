# Experiment Log

## Baseline (original student pipeline, 250m uniform stop radius)

| Metric | Value |
|--------|-------|
| ALL excl walk Jaccard | 0.7683 |
| ALL excl walk exact match | 70.7% |
| CAR→CAR | 82.3% |
| BUS→BUS (trip-level) | 71.2% |
| BUS→BUS (segment-level exact) | 21.7% |
| SUBWAY OOF F1 | 0.66 |

Source: validate_pipeline.py Level 3 + Level 4 output from original code.

---

## Experiment 1: Reduce stop radius 1000m → 50m (uniform)

**Change**: `stop_search_radius_m` from 1000 → 50 in validate_pipeline.py config.

**Result**: Ran, but SUBWAY features collapsed. DURATION_BUS BUS/CAR ratio improved (1.5x → 4.7x at 30m), but METRO features degraded because underground GPS noise means metro stops need wider radius.

**Verdict**: Partial improvement. Need per-mode radii.

---

## Experiment 2: Bus route sequence features (30m radius)

**Change**: New features `BUS_SEQ_STOPS` and `BUS_SEQ_ROUTES` in stops.py. At 30m, BUS=12.7 vs CAR=6.4 unique stops (1.98x ratio).

**Status**: Feature is computed but classifier has NOT been retrained with it yet (model loaded from cache every time).

**Verdict**: PENDING — awaiting clean retrain.

---

## Experiment 3: Segment-level training + walking relabeling

**Change**: In fit() Step 5, PELT-segment training trips, relabel start/end segments with SPEED_AVG < 2 m/s as WALKING. Train on segments instead of whole trips.

**Hypothesis**: "Single-modal" training trips include walking access/egress legs, diluting the classifier's learned features.

**Status**: Code added but NOT executed — model loaded from cache.

**Verdict**: PENDING — awaiting clean retrain.

---

## Experiment 4: Per-mode stop radii (50m default, 250m metro)

**Change**: `metro_stop_radius_m=250` in config. BallTree queries at 250m, then filters per-mode in `_process_chunk`. Bus/tram/rail use 50m, metro uses 250m.

**Hypothesis**: Preserves SUBWAY feature quality while keeping tight radii for other modes.

**Status**: Code pushed, NOT yet run.

**Verdict**: PENDING — run with `python validate_pipeline.py --clean`.

---

## Experiment 5: Post-processing (CAR→BUS reclassification)

**Change**: postprocessing.py with 3 rules: CAR→BUS based on bus evidence, sandwich repair, segment merging.

**Result**: Rule 1 reclassified 28,683 segments CAR→BUS — far too aggressive. CAR→CAR dropped to 21.5%.

**Verdict**: FAILED. Disabled (`postprocess_enabled=False`).

---

## Next run

Experiments 2, 3, 4 all need a clean retrain to verify. Run:
```bash
git pull fork main
python validate_pipeline.py --clean
```

Compare against baseline numbers above.
