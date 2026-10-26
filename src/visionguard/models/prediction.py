"""One checkpoint-faithful path for inference, ranking, calibration and decisions."""
import time
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


def predict_checkpoint(detector, image, name):
    started = time.perf_counter()
    try:
        if not detector.is_loaded or detector.model is None:
            raise RuntimeError(f"Load a trained {name} category checkpoint before inference.")
        rgb, pil = detector.preprocessor.load_and_validate(image)
        tensor = to_tensor(pil).unsqueeze(0).to(detector.device)
        model = detector.model
        with torch.inference_mode():
            inputs = model.pre_processor(tensor) if model.pre_processor else tensor
            raw = model.model(inputs)
            calibrated = model.post_processor(raw) if model.post_processor else raw
        raw_score = float(raw.pred_score.detach().cpu().item())
        score = float(calibrated.pred_score.detach().cpu().item())
        if raw.anomaly_map is None or calibrated.anomaly_map is None:
            raise RuntimeError("The detector did not return a spatial anomaly map.")
        h, w = rgb.shape[:2]
        raw_map = cv2.resize(raw.anomaly_map.detach().cpu().squeeze().numpy().astype(np.float32), (w, h))
        anomaly_map = cv2.resize(calibrated.anomaly_map.detach().cpu().squeeze().numpy().astype(np.float32), (w, h))
        if not (np.isfinite(raw_score) and np.isfinite(score) and np.isfinite(raw_map).all()
                and np.isfinite(anomaly_map).all()):
            raise ValueError("Non-finite model output; check checkpoint calibration.")
        band = max(0.0, (detector.review_high - detector.review_low) / 2)
        label = ("LIKELY_ANOMALOUS" if score >= detector.threshold + band else
                 "LIKELY_NORMAL" if score < detector.threshold - band else "REVIEW_REQUIRED")
        return dict(model_name=name, image_score=score, raw_score=raw_score,
                    anomaly_map=anomaly_map, raw_anomaly_map=raw_map, predicted_label=label,
                    threshold=detector.threshold, inference_ms=(time.perf_counter() - started) * 1000, error=None)
    except Exception as exc:
        return dict(model_name=name, image_score=None, anomaly_map=None, predicted_label="REVIEW_REQUIRED",
                    threshold=detector.threshold, inference_ms=(time.perf_counter() - started) * 1000, error=str(exc))
