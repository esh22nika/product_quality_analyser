# Completion and verification record

## Implemented and exercised

- Supported isolated runtime: torch 2.10.0+cpu, torchvision 0.25.0+cpu,
  Anomalib 2.7.0, Gradio 6.18.0. Global PyTorch was not replaced.
- 29 automated tests pass; result file: `artifacts/pytest_release_final.xml`.
- Real trained PatchCore and EfficientAD inference, public image examples,
  image ZIP export with real score arrays, partial batch failure handling,
  missing-mask checks and persistent SQLite inspection/feedback.
- Corrected schema-v2 image/pixel evaluation and a working comparison CLI.
  Reports distinguish raw ranking scores and calibrated decision thresholds;
  normal images count toward localization false positives. Legacy metric reports
  are excluded from the UI because their IoU/Dice implementation was incorrect.
- Actual image smoke comparisons on identical eligible samples: 10 MVTec bottle
  images and four VisA pcb1 images for each model. This is **not** a full-test run.
  Reports: `reports/release_image_comparison.csv`, individual `verified_*.json` files.
- Small brightness/blur robustness sweeps, with measured AUROC changes saved to
  `reports/release_robustness_patchcore.csv` and `release_robustness_efficientad.csv`.
- Verified downloadable/licensed real conveyor footage with attribution and a
  checksum; separately labeled synthetic benchmark-image sequence. No recordings
  or original photographs required.
- Actual trained-model processing of both prepared video sources, H.264 outputs,
  sampled-frame heatmaps/maps, timelines, galleries, CSVs and JSON provenance.
- Gradio HTTP/API roundtrip for public-video inspection and model-comparison
  reports; nine video outputs returned including playable video and downloadable
  report files. Evidence: `artifacts/gradio_release_verification.json`.
- Annotation-gated frame/event metrics, one-to-one temporal matching, confirmation
  delay, minimum-consecutive evidence and duplicate-alert distinction tested with
  small deterministic fixtures. No public-video annotations were invented.
- Generated source/hash/evaluation cards for all 43 existing category checkpoints
  under `docs/model_cards/`; empty measurement lists mean not evaluated locally.

## Observed limits and deferred work

- The real conveyor clip is unlabeled and out of domain for the existing category
  weights. Its results are qualitative only; video accuracy metrics are null.
- The synthetic video is not recorded product motion. Its watermark is displayed
  in a banner excluded from inference, and it cannot be used as a real-video benchmark.
- Small subset results are insufficient to establish overall quality. In particular,
  EfficientAD pcb1 had zero pixel IoU at the fixed 0.5 threshold on the four-image
  smoke subset. This result is retained, not hidden or replaced by publisher numbers.
  Use a suitable separate validation split to investigate thresholds; do not tune
  on the test images and present the result as an untouched evaluation.
- Third-party model selection/calibration may have used benchmark test samples;
  these existing checkpoints do not establish a matched-training experiment.
- Full all-category and full robustness sweeps, optional MVTec AD 2, object tracking,
  high-resolution tiling and supervised segmentation are not claimed completed.
  The documented evaluation CLI can run the full eligible image lists when desired.
- The new runtime reused installed Python dependencies through a local virtual
  environment; an independent clean-machine installation was not performed.

## Training / two-hour CPU constraint

No training was performed. Existing category checkpoints suffice for the completed
image and demonstration-video workflows. A new conveyor-scene EfficientAD detector
requires normal frames from a suitable public dataset and may exceed two hours on
CPU. It was not launched. A GPU/pilot timing is required before promising a budget.

Start the updated demo with `.venv/Scripts/python.exe app.py` from the project root,
then open http://127.0.0.1:7860. Exact setup, preparation, evaluation, training and
test commands are documented in README.md and the linked guides.
