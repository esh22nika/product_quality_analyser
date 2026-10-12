import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd

from visionguard.models.loader import ModelManager
from visionguard.inference.postprocessing import AnomalyPostprocessor
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.inference.video")

class VideoInspectionPipeline:
    """Processes video frames, aggregates temporal events, and produces annotated video."""

    def __init__(
        self,
        model_name: str = "patchcore",
        category: Optional[str] = "bottle",
        sampling_interval: int = 2,
        decision_threshold: float = 0.5,
        consecutive_frames: int = 3,
        merge_gap_frames: int = 5,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.category = category
        self.sampling_interval = max(1, sampling_interval)
        self.decision_threshold = decision_threshold
        self.consecutive_frames = consecutive_frames
        self.merge_gap_frames = merge_gap_frames
        self.device = device
        self.postprocessor = AnomalyPostprocessor()

    def process_video(
        self,
        video_path: str | Path,
        output_video_path: Optional[str | Path] = None,
        max_duration_s: float = 60.0
    ) -> Dict[str, Any]:
        """Runs frame extraction, model inference, temporal aggregation, and creates annotated video."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0.0

        max_frames_to_read = int(min(total_frames, fps * max_duration_s))

        model = ModelManager.get_model(
            model_name=self.model_name,
            category=self.category,
            device=self.device
        )
        model.threshold = self.decision_threshold

        frame_records: List[Dict[str, Any]] = []
        suspicious_frames_gallery: List[Dict[str, Any]] = []

        writer = None
        if output_video_path is not None:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Use mp4v codec for cross-platform compatibility
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (orig_w, orig_h))

        frame_idx = 0
        processed_count = 0
        start_processing_time = time.time()

        while cap.isOpened() and frame_idx < max_frames_to_read:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            is_sampled = (frame_idx % self.sampling_interval == 0)

            if is_sampled:
                processed_count += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pred = model.predict(frame_rgb)
                score = pred.get("image_score", 0.0)
                is_anomalous = bool(score >= self.decision_threshold)
                anomaly_map = pred.get("anomaly_map")

                frame_records.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": round(timestamp, 2),
                    "anomaly_score": score,
                    "is_anomalous": is_anomalous,
                    "predicted_label": pred.get("predicted_label", "REVIEW_REQUIRED")
                })

                if is_anomalous and anomaly_map is not None and len(suspicious_frames_gallery) < 10:
                    heatmap = self.postprocessor.generate_heatmap(anomaly_map)
                    overlay = self.postprocessor.generate_overlay(frame_rgb, heatmap)
                    suspicious_frames_gallery.append({
                        "frame_idx": frame_idx,
                        "timestamp_s": round(timestamp, 2),
                        "score": score,
                        "overlay": overlay
                    })

                # Visual annotation for output video
                if writer is not None:
                    annotated_bgr = frame_bgr.copy()
                    color = (0, 0, 255) if is_anomalous else (0, 255, 0)
                    status_text = f"Frame {frame_idx} | Score: {score:.3f} | {'ALERT' if is_anomalous else 'NORMAL'}"
                    cv2.putText(annotated_bgr, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    writer.write(annotated_bgr)
            else:
                if writer is not None:
                    # Write frame with last known status indicator
                    writer.write(frame_bgr)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        total_time = time.time() - start_processing_time
        fps_throughput = processed_count / total_time if total_time > 0 else 0.0

        # Temporal aggregation of anomalous frames into events
        df_frames = pd.DataFrame(frame_records)
        events = self._aggregate_events(df_frames, fps)
        df_events = pd.DataFrame(events)

        return {
            "metadata": {
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "fps": round(fps, 2),
                "duration_s": round(duration_s, 2),
                "resolution": f"{orig_w}x{orig_h}",
                "processing_time_s": round(total_time, 2),
                "throughput_fps": round(fps_throughput, 1)
            },
            "frame_dataframe": df_frames,
            "event_dataframe": df_events,
            "suspicious_gallery": suspicious_frames_gallery,
            "output_video_path": str(output_video_path) if output_video_path else None
        }

    def _aggregate_events(self, df_frames: pd.DataFrame, fps: float) -> List[Dict[str, Any]]:
        """Groups nearby anomalous frames into unified defect events."""
        if df_frames.empty:
            return []

        anomalous_frames = df_frames[df_frames["is_anomalous"] == True].copy()
        if anomalous_frames.empty:
            return []

        events = []
        current_event_frames = []

        for _, row in anomalous_frames.iterrows():
            f_idx = row["frame_idx"]
            if not current_event_frames:
                current_event_frames.append(row)
            else:
                prev_idx = current_event_frames[-1]["frame_idx"]
                if (f_idx - prev_idx) <= (self.sampling_interval * self.merge_gap_frames):
                    current_event_frames.append(row)
                else:
                    if len(current_event_frames) >= self.consecutive_frames:
                        events.append(self._summarize_event(current_event_frames, fps))
                    current_event_frames = [row]

        if len(current_event_frames) >= self.consecutive_frames:
            events.append(self._summarize_event(current_event_frames, fps))

        return events

    def _summarize_event(self, event_frames: List[pd.Series], fps: float) -> Dict[str, Any]:
        """Summarizes a cluster of suspicious frames."""
        start_f = event_frames[0]["frame_idx"]
        end_f = event_frames[-1]["frame_idx"]
        scores = [f["anomaly_score"] for f in event_frames]
        return {
            "event_id": f"EVT_{start_f}_{end_f}",
            "start_time_s": round(start_f / fps, 2),
            "end_time_s": round(end_f / fps, 2),
            "duration_s": round((end_f - start_f + 1) / fps, 2),
            "suspicious_frame_count": len(event_frames),
            "peak_anomaly_score": round(float(np.max(scores)), 4),
            "mean_anomaly_score": round(float(np.mean(scores)), 4)
        }
