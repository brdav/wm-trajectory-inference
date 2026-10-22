# QC validation (how good is the QC?)

All numbers 2026-10-21, measured with two analysis scripts that were removed afterwards (last
version: `git show 61e6655:scripts/qc_validation/qc_depth_spotcheck.py` and `.../qc_placement.py`;
the raw clip lists, scores and figures were not kept, rerunning needs a GPU run, see "Reproduce").
The only independent reference for a clip's scale is the depth-based one (DA3METRIC-LARGE vs the
SLAM point cloud), so every statement below is "the two estimates agree / clearly disagree" (ratio
depth/GPS in 0.8-1.25 / outside 0.67-1.5), never ground truth. A disagreement does not say which
estimate is wrong; in some clips it was the depth scale.

- **The reference is usable.** On a random sample of 363 unflagged `use_fused` clips with a
  confident depth scale the depth/GPS ratio has median 0.98 and a robust spread of 8.3% (these
  bands are about 2 and 4 sigma).
- **Unflagged clips (105,063).** 6 of 363 (1.7%, 95% CI 0.8-3.6%) clearly disagree, i.e. at most
  ~1,700 clips (800-3,700) may have a wrong scale that QC cannot see; another 3.3% are ambiguous
  (0.67-0.8 / 1.25-1.5). 9% of the sample had no confident depth scale and is not covered.
- **QC mostly errs by passing, not rejecting.** Among clips QC rejected for the scale check, 74 of
  248 with a confident depth scale (30%, CI 24-36%) have a depth scale that agrees with the GPS
  one: probable false positives (~70-100 clips), against up to ~1,700 clips passed that disagree.
- **Where the cross-check tier pays off** (clearly contradicted): fused/GPS length ratio 1.5-2
  (now a hard reject): 17 of 40; ratio < 0.8: 21% (184 of 881); no ratio computable: 17% (102 of
  583); ratio 1.25-1.5: 18% (33 of 188); mean map-match error > 10 m: 4% (102 of 2,843; flat across
  10-15 / 15-25 / > 25 m; an earlier spot check suggested 37% but was confounded by the other
  edge-case clips). Speed 30-50 m/s and SLAM step ratio 25-50 (near-threshold) were fine (0 of 40,
  1 of 30).
- **Placement is fine; timing is not always.** The distance from raw GPS fixes to the fused path
  (cross-track, insensitive to timing) has median 3.1 m per clip; only 10 of 1,496 QC-passed clips
  exceed 20 m and 2 exceed 50 m. The fused path at the *same time* as a fix is > 50 m off in 42
  clips, but those lie 5 m from the path: an implied GPS-vs-video time offset of a median 3 s (p90
  9 s) in ~3% of clips. Not a pose error, but beware when using GPS timestamps.
- **Cross-check outcome (4,495 cross-checked clips):** 50% confirmed by the depth scale, 17-21%
  contradicted for the length-ratio triggers but only 4% for the map-matching trigger (102 of
  2,843, all of which were then overruled by the GPS track length); noisy depth estimates (about a
  quarter) are judged on their interquartile range and only 32 clips remain unverifiable.
- **Limits:** one sample, one pool, depth scale +-8-10%; the rates above were measured on
  confident depth estimates only (9% of the sample had noisy ones and are not covered); `static`
  and `too_few_keyframes` clips were checked against GPS extent only, not against depth.
- **Depth scale accuracy:** validated on 5 clips with GPS: within about +-8% of the GPS-derived
  scale, no systematic bias. Treat as ~+-10%.

## Reproduce

Draw a stratified sample of clips into a `clip_list.txt` (dataset-relative `.mp4` path, TAB,
stratum label; the strata and sample sizes are in the spot-check script in git history, see
above), score it with `scripts/estimate_metric_scale.py --clip_list ... --pipeline_dir <records
dir> --work_dir <work dir>` (about 25 clips/min on a 4-GPU node), then compare each clip's depth
scale with its GPS-fused scale (the spot-check script does this; its placement companion needs no
GPU).
