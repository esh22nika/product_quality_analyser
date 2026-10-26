"""Reproducible image evaluation; no training or test-set threshold selection."""
import csv
import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from .image_metrics import ImageMetricsCalculator
from .pixel_metrics import PixelMetricsCalculator
from .reports import BenchmarkReporter
from .region_metrics import aupro


def samples(root, dataset, category):
    root = Path(root)
    records = []
    if dataset == "visa":
        with (root / "split_csv/1cls.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["object"] == category and row["split"] == "test":
                    label = int(row["label"].lower() != "normal")
                    records.append((root / row["image"], root / row["mask"] if label else None, label))
    else:
        base = root / category
        for image in sorted((base / "test").glob("*/*")):
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            label = int(image.parent.name != "good")
            records.append((image, base / "ground_truth" / image.parent.name / f"{image.stem}_mask.png" if label else None, label))
    if not records:
        raise FileNotFoundError(f"No test samples for {dataset}/{category} under {root}")
    for image, mask, label in records:
        if not image.is_file() or (label and (mask is None or not mask.is_file())):
            raise FileNotFoundError(f"Missing test image or defect mask: {image}, {mask}")
    return records


def evaluate(model_name, dataset, category, root, reports="./reports", limit_per_class=None, threshold=0.5, corruption="none"):
    from visionguard.models.loader import ModelManager
    records = samples(root, dataset, category)
    if limit_per_class:
        records = [
            record for label in [0, 1] for record in [r for r in records if r[2] == label][:limit_per_class]]
    model = ModelManager.get_model(model_name, category)
    model.threshold = threshold
    truth, ranking, calibrated, masks, maps, decisions, latencies, paths = [], [], [], [], [], [], [], []
    started = time.perf_counter()
    for image, mask_path, label in records:
        value = str(image)
        if corruption != "none":
            from .robustness import RobustnessEvaluator
            value = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
            transforms = {"brightness": lambda x: RobustnessEvaluator.apply_brightness(x, .7),
                          "contrast": lambda x: RobustnessEvaluator.apply_contrast(x, .7),
                          "blur": lambda x: RobustnessEvaluator.apply_blur(x, 5),
                          "jpeg": lambda x: RobustnessEvaluator.apply_jpeg_compression(x, 40)}
            if corruption not in transforms:
                raise ValueError(f"Unknown corruption: {corruption}")
            value = transforms[corruption](value)
        prediction = model.predict(value)
        if prediction.get("error"):
            raise RuntimeError(f"Inference failed on {image}: {prediction['error']}")
        truth.append(label)
        ranking.append(prediction["raw_score"])
        calibrated.append(prediction["image_score"])
        latencies.append(prediction["inference_ms"])
        paths.append(str(image))
        # Include normal-image zero masks: omitting them inflates localization metrics.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if label else np.zeros(prediction["anomaly_map"].shape, np.uint8)
        if mask is None:
            raise ValueError(f"Unreadable defect mask: {mask_path}")
        masks.append(cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST))
        maps.append(cv2.resize(prediction["raw_anomaly_map"], (256, 256)))
        decisions.append(cv2.resize(prediction["anomaly_map"], (256, 256)))
    image_metrics = ImageMetricsCalculator.calculate(truth, ranking, threshold, decision_scores=calibrated)
    pixel_metrics = PixelMetricsCalculator.calculate(masks, maps, threshold, decision_maps=decisions)
    pixel_metrics.update(aupro_0_3_approx=aupro(masks, maps), aupro_threshold_count=64,
                         aupro_protocol="64 quantile thresholds, normalized partial PRO AUC at FPR <= 0.3")
    metrics = {**image_metrics, **pixel_metrics, "mean_inference_ms": float(np.mean(latencies)),
               "median_inference_ms": float(np.median(latencies)), "elapsed_s": time.perf_counter() - started}
    manifest = "\n".join(paths)
    import torch
    info = dict(name=model_name, device=model.device, hardware=platform.processor(),
                dependencies={p: importlib.metadata.version(p) for p in ["torch", "torchvision", "anomalib", "lightning"]},
                checkpoint=str(getattr(model, "checkpoint_path", "unknown")),
                torch_threads=torch.get_num_threads(), score_protocol="raw ranking; checkpoint calibration for threshold decisions",
                training_provenance="existing category checkpoint; third-party validation/test selection may be unknown")
    checkpoint_path = Path(info["checkpoint"])
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as stream:
            info["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    data = dict(name=dataset, category=category, root=str(Path(root).resolve()),
                split="official test" if not limit_per_class else "deterministic balanced official-test subset",
                limit_per_class=limit_per_class, image_paths=paths,
                sample_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                pixel_evaluation_resolution=[256, 256], corruption=corruption,
                threshold_selection="fixed before evaluation; no test tuning",
                comparison_limitations="Same inference samples and code; existing checkpoints may differ in training/validation protocol.")
    experiment = f"verified_{model_name}_{dataset}_{category}_{'full' if not limit_per_class else limit_per_class}_{corruption}"
    path = BenchmarkReporter(reports).save_experiment_report(experiment, metrics, info, data)
    ModelManager.clear_cache()
    return path, metrics
