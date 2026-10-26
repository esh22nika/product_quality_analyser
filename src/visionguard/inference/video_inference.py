"""Bounded streaming video inspection with explicit provenance and annotation gating."""
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from visionguard.models.loader import ModelManager
from visionguard.inference.postprocessing import AnomalyPostprocessor
from visionguard.evaluation.video_metrics import VideoMetricsCalculator

FRAME_COLUMNS = ["frame_idx", "timestamp_s", "anomaly_score", "raw_score", "is_anomalous", "predicted_label"]
EVENT_COLUMNS = ["event_id", "start_time_s", "end_time_s", "detected_time_s", "duration_s", "suspicious_frame_count", "peak_anomaly_score", "mean_anomaly_score"]

class VideoInspectionPipeline:
    def __init__(self, model_name="patchcore", category="bottle", sampling_interval=15,
                 decision_threshold=0.5, consecutive_frames=3, merge_gap_frames=5,
                 device="auto", min_event_duration_s=0.0, roi=None, max_analysis_frames=120):
        if sampling_interval < 1 or consecutive_frames < 1 or merge_gap_frames < 0 or max_analysis_frames < 1:
            raise ValueError("Sampling/event counts must be positive and merge gap nonnegative")
        self.model_name, self.category, self.device = model_name, category, device
        self.sampling_interval = int(sampling_interval)
        self.decision_threshold = float(decision_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.merge_gap_frames = int(merge_gap_frames)
        self.min_event_duration_s = max(0, float(min_event_duration_s))
        self.roi, self.max_analysis_frames = roi, int(max_analysis_frames)
        self.postprocessor = AnomalyPostprocessor()

    def process_video(self, video_path, output_video_path=None, max_duration_s=60.0, provenance=None, annotations=None, on_progress=None):
        def report(fraction, message):
            if on_progress is not None:
                on_progress(fraction, message)

        report(0, "Opening video…")
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".webm"}:
            raise ValueError("Use MP4, AVI or MOV (prepared WebM is also supported)")
        if max_duration_s <= 0 or video_path.stat().st_size > 2 * 1024**3:
            raise ValueError("Duration must be positive; uploaded video must be under 2 GiB")
        cap = cv2.VideoCapture(str(video_path))
        writer = None
        records, gallery = [], []
        out = Path(output_video_path) if output_video_path else None
        started = time.perf_counter()
        from visionguard.config import load_default_config
        limits = load_default_config()
        try:
            if not cap.isOpened():
                raise ValueError("Video cannot be decoded. Convert to H.264 MP4 using FFmpeg.")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or not 0 < fps <= 1000 or min(width, height) <= 0 or max(width, height) > limits.get("video.max_dimension", 4096):
                raise ValueError("Invalid/unsupported video metadata")
            suggested = (provenance or {}).get("recommended_roi") if (provenance or {}).get("sequence_type") == "synthetic_image_sequence" else None
            roi = tuple(map(int, self.roi if self.roi is not None else suggested)) if self.roi is not None or suggested is not None else (0, 0, width, height)
            if len(roi) != 4:
                raise ValueError("ROI must be x,y,width,height in source pixels")
            x, y, rw, rh = roi
            if x < 0 or y < 0 or min(rw, rh) <= 0 or x + rw > width or y + rh > height:
                raise ValueError("ROI is outside the source frame")
            report(0.02, f"Loading {self.model_name} / {self.category} checkpoint…")
            model = ModelManager.get_model(self.model_name, self.category, device=self.device)
            model.threshold = self.decision_threshold
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                output_width = max(2, min(width, limits.get("video.output_max_width", 960)) // 2 * 2)
                output_height = max(2, int(height * output_width / width) // 2 * 2)
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
                if not writer.isOpened():
                    raise RuntimeError("MP4 encoder unavailable. Install FFmpeg-enabled OpenCV.")
            frames_read = 0
            limit = int(fps * min(max_duration_s, limits.get("video.max_duration_s", 60)))
            expected = min(self.max_analysis_frames, math.ceil(min(total, limit) / self.sampling_interval)) if total > 0 else self.max_analysis_frames
            report(0.05, f"Analyzing up to {expected} sampled frames on {getattr(model, 'device', self.device)}. CPU inference can take several seconds per frame.")
            while frames_read < limit and len(records) < self.max_analysis_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                index = frames_read
                frames_read += 1
                text, color = "NOT ANALYZED (sampling)", (180, 180, 180)
                if index % self.sampling_interval == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    crop = rgb[y:y+rh, x:x+rw]
                    pred = model.predict(crop)
                    if pred.get("error") or pred.get("anomaly_map") is None:
                        raise RuntimeError(f"Frame {index}: {pred.get('error', 'No anomaly map')}")
                    score = float(pred["image_score"])
                    if not math.isfinite(score):
                        raise ValueError(f"Non-finite frame score at {index}")
                    suspicious = score >= self.decision_threshold
                    records.append(dict(frame_idx=index, timestamp_s=index / fps, anomaly_score=score,
                                        raw_score=float(pred.get("raw_score", score)), is_anomalous=suspicious,
                                        predicted_label=pred["predicted_label"]))
                    heatmap = self.postprocessor.generate_heatmap(pred.get("raw_anomaly_map", pred["anomaly_map"]), scale="relative")
                    overlay = self.postprocessor.generate_overlay(crop, heatmap)
                    rgb[y:y+rh, x:x+rw] = overlay
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(bgr, (x, y), (x+rw-1, y+rh-1), (255, 255, 0), 1)
                    text = f"{index/fps:.2f}s | score {score:.3f} | {'SUSPICIOUS' if suspicious else 'BELOW THRESHOLD'}"
                    color = (0, 0, 255) if suspicious else (0, 200, 0)
                    if out:
                        maps = out.parent / "maps"
                        maps.mkdir(exist_ok=True)
                        np.savez_compressed(maps / f"frame_{index:08d}.npz", timestamp_s=index/fps,
                                            calibrated=cv2.resize(pred["anomaly_map"], (256, 256)),
                                            raw=cv2.resize(pred.get("raw_anomaly_map", pred["anomaly_map"]), (256, 256)), roi=np.array(roi))
                    if len(gallery) < 12:
                        thumb = cv2.resize(rgb, (min(width, 480), max(1, int(height * min(width, 480) / width))))
                        gallery.append(dict(frame_idx=index, timestamp_s=index/fps, score=score, overlay=thumb))
                    report(0.05 + 0.8 * len(records) / max(1, expected),
                           f"Analyzed {len(records)}/{expected} sampled frames · video {index/fps:.1f}s · score {score:.3f}")
                mode = "SYNTHETIC | " if (provenance or {}).get("sequence_type") == "synthetic_image_sequence" else ""
                # Keep the synthetic-source banner readable at the top of the frame.
                caption = mode + text
                font_scale = min(0.45, max(0.2, (width - 20) / max(1, len(caption) * 10)))
                cv2.rectangle(bgr, (0, max(0, height-30)), (width, height), (15, 20, 26), -1)
                cv2.putText(bgr, caption, (10, max(12, height-10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                if writer:
                    writer.write(cv2.resize(bgr, (output_width, output_height)))
            if not records:
                raise ValueError("No frames could be analyzed")
        except Exception:
            if writer is not None:
                writer.release()
                writer = None
            if out:
                out.unlink(missing_ok=True)
            raise
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        elapsed = time.perf_counter() - started
        encoding_started = time.perf_counter()
        if out:
            from visionguard.inference.video_encoding import browser_encode
            report(0.9, "Encoding browser-playable video…")
            browser_encode(out)
        df = pd.DataFrame(records, columns=FRAME_COLUMNS)
        events = self._aggregate_events(df, fps)
        for event in events:
            event["end_time_s"] = min(event["end_time_s"], frames_read / fps)
            event["duration_s"] = event["end_time_s"] - event["start_time_s"]
        event_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
        origin = provenance or dict(sequence_type="uploaded_video", annotation_type="none")
        metrics = VideoMetricsCalculator.calculate()
        if annotations is not None:
            if origin.get("sequence_type") == "synthetic_image_sequence":
                raise ValueError("Synthetic demonstrations cannot be reported as real-video benchmarks")
            if annotations.get("annotation_type") != "temporal_intervals" or annotations.get("complete") is not True:
                raise ValueError("Evaluation needs complete temporal_intervals annotations; partial/unlabeled video is qualitative")
            end = frames_read / fps
            if annotations.get("coverage_start_s", 0) > 0 or annotations.get("coverage_end_s", -1) < end:
                raise ValueError("Annotations do not cover the complete processed video interval")
            truth = annotations.get("events")
            if not isinstance(truth, list):
                raise ValueError("Annotations must provide an events list, including [] for a verified normal clip")
            for event in truth:
                if not (0 <= event["start_time_s"] < event["end_time_s"] <= annotations["coverage_end_s"]):
                    raise ValueError("Invalid ground-truth event interval")
            truth = [dict(start_time_s=e["start_time_s"], end_time_s=min(e["end_time_s"], end)) for e in truth if e["start_time_s"] < end]
            labels = [int(any(e["start_time_s"] <= r["timestamp_s"] < e["end_time_s"] for e in truth)) for r in records]
            metrics = VideoMetricsCalculator.calculate(labels, df.raw_score.tolist(), events, truth)
        metadata = dict(total_frames=total, processed_frames=len(records), frames_read=frames_read, fps=fps,
                        duration_s=total/fps if total > 0 else None, processed_duration_s=frames_read/fps,
                        resolution=f"{width}x{height}", processing_time_s=elapsed, throughput_fps=len(records)/elapsed,
                        truncated=(total > frames_read), sampling_interval=self.sampling_interval, roi=list(roi),
                        model=self.model_name, category=self.category, tracking="none; events are temporal clusters, not product counts",
                        checkpoint=getattr(model, "checkpoint_path", None), device=getattr(model, "device", self.device),
                        output_codec="h264/yuv420p", encoding_time_s=time.perf_counter()-encoding_started,
                        saved_map_resolution=[256, 256],
                        heatmap_visualization="relative raw-map contrast per frame; colors are not defect probabilities; scores and events use calibrated values",
                        unsampled_frames="No prediction; marked NOT ANALYZED", provenance=origin)
        if out:
            df.to_csv(out.parent / "frame_scores.csv", index=False)
            event_df.to_csv(out.parent / "inspection_events.csv", index=False)
            inspection_report = dict(metadata=metadata, metrics=metrics, events=events, annotations=annotations)
            (out.parent / "inspection_report.json").write_text(json.dumps(inspection_report, indent=2, allow_nan=False), encoding="utf-8")
        report(1, f"Completed: {len(records)} sampled frames; {len(events)} inspection events.")
        return dict(metadata=metadata, frame_dataframe=df, event_dataframe=event_df, suspicious_gallery=gallery,
                    output_video_path=str(out) if out else None, metrics=metrics)

    def _aggregate_events(self, df_frames, fps):
        if df_frames.empty:
            return []
        # First require a genuine consecutive run; isolated positives cannot qualify through gap merging.
        runs, current = [], []
        for _, row in df_frames.iterrows():
            if bool(row["is_anomalous"]):
                current.append(row)
            else:
                if len(current) >= self.consecutive_frames:
                    runs.append(current)
                current = []
        if len(current) >= self.consecutive_frames:
            runs.append(current)
        merged = []
        for run in runs:
            if merged and run[0]["frame_idx"] - merged[-1][-1]["frame_idx"] <= self.sampling_interval * (self.merge_gap_frames + 1):
                merged[-1].extend(run)
            else:
                merged.append(run)
        result = [self._summarize_event(run, fps) for run in merged]
        return [event for event in result if event["duration_s"] >= self.min_event_duration_s]

    def _summarize_event(self, frames, fps):
        start, end = int(frames[0]["frame_idx"]), int(frames[-1]["frame_idx"])
        scores = [float(frame["anomaly_score"]) for frame in frames]
        return dict(event_id=f"EVT_{start}_{end}", start_time_s=start/fps, end_time_s=(end+self.sampling_interval)/fps,
                    detected_time_s=float(frames[min(self.consecutive_frames, len(frames))-1]["frame_idx"])/fps,
                    duration_s=(end-start+self.sampling_interval)/fps, suspicious_frame_count=len(frames),
                    peak_anomaly_score=max(scores), mean_anomaly_score=float(np.mean(scores)))
