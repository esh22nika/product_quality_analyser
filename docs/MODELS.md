# Model use and training limits

The application uses local PyTorch/Anomalib inference, not an external paid API.
PatchCore and EfficientAD share checkpoint-faithful preprocessing/postprocessing.
EfficientAD performs ImageNet normalization inside its network; it must not be
applied twice. Calibration tensors and teacher feature statistics are retained.
Visualization colors are not probabilities. Thresholds are operational cutoffs.

## Existing checkpoints

- `checkpoints/patchcore_<category>.pt`: existing locally trained category models.
- `checkpoints/public_efficientad/<category>/model.ckpt`: pinned public trained
  category detectors, plus per-file source, revision and SHA-256 records.
- `docs/model_cards/index.json`: generated cards for each existing checkpoint,
  including checkpoint hashes, local measured reports and provenance limitations.

EfficientAD MVTec source: https://huggingface.co/micguida1/mvtec-anomaly-checkpoints
and pcb1 source: https://huggingface.co/thangkt/visionqc-efficientad-medium-pcb1
The pcb1 publisher used a batched medium-model variant. It is compatible for
inference but is not a reproduction of the original batch-size-one training protocol.

`python scripts/download_public_efficientad.py` downloads the 16 category checkpoints
(about 1.31 GB). No training is performed. Models load only when needed; at most two
model instances are cached. Missing or incompatible weights fail explicitly.

## Reproducible training entry points (not run during completion)

```powershell
python scripts/train.py --model patchcore --dataset mvtec_ad --category bottle --root mvtec_anomaly_detection
python scripts/train.py --model efficientad --dataset mvtec_ad --category bottle --root mvtec_anomaly_detection
```

EfficientAD uses training batch size one and normal training images. Its validation
split is held out from test images, with seed 42; those validation images are not
used as normal training images. Do not treat metrics on the original complete test
set as an untouched holdout after model/threshold selection on part of that set.
For sustained training use the corrected Colab notebook and preserve resumable Drive
checkpoints. Existing checkpoints do not need retraining for image/video inference.

**A new scene-specific conveyor EfficientAD model may take more than two hours on
CPU. No such training has been launched.** Use a GPU and a measured pilot estimate
before committing to a full new-category run. No accurate CPU duration can be
guaranteed from the current inference-only timings.
