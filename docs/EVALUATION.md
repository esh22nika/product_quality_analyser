# Evaluation protocol v2

Legacy reports were produced with a pixel-threshold bug (thresholding truth instead
of predictions). They must not be cited. The UI excludes reports without schema v2.
Files remain on disk for audit; regenerating corrected reports does not delete them.

```powershell
python scripts/evaluate.py --model efficientad --dataset mvtec_ad --category bottle --root mvtec_anomaly_detection
python scripts/benchmark.py --categories bottle cable --root mvtec_anomaly_detection
python scripts/benchmark.py --dataset visa --categories pcb1 --root datasets/visa
# Quick balanced smoke comparison; NOT full-test benchmark results:
python scripts/benchmark.py --categories bottle --limit-per-class 5
```

Both architectures use identical deterministic test-image lists, underlying model
resize/preprocessing and the same code. Normal test images contribute zero masks;
every defective image requires its actual mask. Missing samples and inference errors
fail rather than being silently skipped. No training or threshold tuning is performed.

Raw image scores feed AUROC/AP; checkpoint-calibrated scores at a fixed 0.5 threshold
feed precision/recall/F1 and confusion matrices. Likewise pixel ranking uses raw maps,
while segmentation uses calibrated maps. No per-image min/max normalization is used
for metrics. Display colormaps are separate from ranking and calibration.

The image UI defaults to relative display contrast (1st–99th percentile of the raw
map) so narrow calibrated ranges do not appear uniformly green. This changes only
heatmap/overlay colors. Scores, calibrated threshold masks and all metrics are
unchanged. A fixed calibrated 0–1 color option remains available; relative red
means a higher response within that image, not a defect probability.

Pixel evaluation downsamples maps to 256 x 256 and masks with nearest-neighbor
interpolation. Pixel AUROC, AP, IoU and Dice are reported, with normal-image false
positives included. AP uses at most 500,000 deterministic pixels (seed 42); sample
counts and resolution are recorded. These conventions differ from full-resolution
official evaluation. `aupro_0_3_approx` is the normalized per-region-overlap area
up to false-positive rate 0.3, approximated with 64 raw-score quantile thresholds.
Regions receive equal weight. This finite-threshold approximation is explicitly
named as such and is not advertised as the official benchmark implementation.

Reports include dependency versions, CPU/GPU identity, checkpoint path, evaluation
sample paths + their manifest hash, subset limit, thresholds and latency. Existing
third-party EfficientAD checkpoints may have selected epochs/calibration using the
original test split. This is an inference comparison, not a claim of matched training
or untouched test sets. Publisher scores are not substituted for local measurements.

Robustness utilities in `evaluation/robustness.py` transform copies only. Full
corruption sweeps and optional MVTec AD 2 evaluation have not been run for this release.
Video metrics follow the separate annotation policy in VIDEO_PIPELINE.md.
