# Experiment Log

## Baseline (original student pipeline, 250m uniform stop radius)

| Metric | Value |
|--------|-------|
| ALL excl walk Jaccard | 0.7683 |
| ALL excl walk exact match | 70.7% |
| OOF macro F1 | 0.7775 |
| CAR→CAR | 82.3% (Level 4: 69.9%) |
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

## Experiment 5: Post-processing (CAR→BUS reclassification)

**Change**: postprocessing.py with 3 rules: CAR→BUS based on bus evidence, sandwich repair, segment merging.

**Result**: Rule 1 reclassified 28,683 segments CAR→BUS — far too aggressive. CAR→CAR dropped to 21.5%.

**Verdict**: FAILED. Disabled (`postprocess_enabled=False`).

---

## Run 2024-11-21: Experiments 2+3+4 combined (first clean run)

**Changes applied together**:
- Exp 2: Bus route sequence features (`BUS_SEQ_STOPS`, `BUS_SEQ_ROUTES`) at 30m
- Exp 3: Segment-level training + walking relabeling (train on 322K segments instead of 52K trips)
- Exp 4: Per-mode stop radii (50m bus/tram/rail, 250m metro)

**Level 3 results (OOF):**

| Mode | Orig F1 | Ours F1 | Delta |
|------|---------|---------|-------|
| BIKE | 0.8199 | 0.6541 | -0.1658 |
| BUS | 0.6636 | 0.4569 | -0.2067 |
| CAR | 0.9363 | 0.8681 | -0.0681 |
| SUBWAY | 0.6574 | 0.2304 | **-0.4270** |
| TRAIN | 0.8111 | 0.7602 | -0.0509 |
| TRAMWAY | 0.6740 | 0.5045 | -0.1696 |
| WALKING | 0.8799 | 0.8510 | -0.0289 |
| **MACRO** | **0.7775** | **0.6179** | **-0.1596** |

**Level 4 results:**

| Metric | Baseline | This run | Delta |
|--------|----------|----------|-------|
| ALL excl walk Jaccard | 0.7683 | 0.7178 | -0.0505 |
| ALL excl walk exact | 70.7% | 64.0% | -6.7% |
| SM excl walk exact | — | 66.7% | — |
| MM excl walk Jaccard | — | 0.4923 | — |

**Key observations:**
- Every mode got worse. SUBWAY destroyed (F1 0.66 → 0.23).
- 97.5% of single-modal trips falsely segmented (avg 2.99 segments).
- WALKING→SUBWAY confusion 9.5% (was 1.9%).
- CAR→BUS confusion 7.8% (was 0.8%).
- `BUS_SEQ_STOPS` ranked #35 by feature importance (gain=16.1) — useful but not transformative.

**Diagnosis:** Segment-level training is the likely cause. Training on 322K noisy short segments (instead of 52K trips) degrades all modes. PELT over-segments single-modal trips.

**Action:** Revert to trip-level training. Keep per-mode radii + bus seq features.

---

## Next run: Trip-level training + per-mode radii + bus seq features

Reverted segment-level training. Keeping:
- Per-mode stop radii (50m bus/tram/rail, 250m metro)
- Bus route sequence features at 30m
- Post-processing disabled

This isolates whether per-mode radii + bus seq features help on their own.

```bash
git pull fork main
python validate_pipeline.py --clean
```
