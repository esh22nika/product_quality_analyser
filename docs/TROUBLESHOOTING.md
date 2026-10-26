# Troubleshooting

- Missing checkpoint: download public EfficientAD weights, or use an existing
  `checkpoints/patchcore_<category>.pt`; teacher-only weights are insufficient.
- Incompatible checkpoint: do not set `strict=False` to conceal network mismatch.
  EfficientAD's evaluation-only metric accumulators may be excluded across versions;
  network/statistics/calibration tensors remain strict.
- No video examples: run `prepare_commons_conveyor.py` and optionally
  `make_demo_video.py`, then restart the app to refresh the source selector.
- No video accuracy numbers: expected for unlabeled public footage. It must stay
  qualitative. A complete covered annotation file is required for quantitative metrics.
- Browser video does not play: the project uses bundled FFmpeg to encode H.264.
  Ensure `imageio-ffmpeg==0.6.0` is installed. An explicit conversion error preserves
  the intermediate file for investigation.
- CPU is slow: use a shorter duration, a larger sampling interval, a fixed product
  ROI and the small EfficientAD category models. Four CPU threads are used by default.
  Long video decoding is streaming; maps/gallery and analyzed-frame counts are bounded.
- Old evaluation reports disagree: legacy IoU/Dice had a ground-truth threshold bug.
  Only corrected schema-v2 reports appear in the UI. Regenerate rather than citing old files.
- MVTec weights flag an unrelated conveyor: domain mismatch can cause anomalies.
  A score is not a verified defect classification or a defect probability.
- Camera/product motion: static ROI is supported; object tracking and product-level
  deduplication are not implemented. Events count suspicious time intervals only.
