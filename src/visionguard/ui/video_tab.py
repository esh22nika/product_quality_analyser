from pathlib import Path
import json
import uuid
import gradio as gr
import pandas as pd
from visionguard.inference.video_inference import VideoInspectionPipeline
from visionguard.data.categories import INSPECTION_CATEGORIES
from visionguard.data.public_videos import available_videos
from visionguard.storage.repository import InspectionRepository
from visionguard.config import load_default_config


def create_video_tab():
    config = load_default_config()
    sources = available_videos()
    upload_label = "Upload an existing video"
    with gr.Tab("Video Inspection"):
        gr.Markdown("### Public video inspection\nChoose downloaded footage or upload an existing MP4/AVI/MOV. No recording is required. "
                    "Scores measure visual unfamiliarity. Temporal events are not counts of defective products.")
        with gr.Row():
            with gr.Column():
                source = gr.Dropdown(choices=[upload_label] + list(sources), value=next(iter(sources), upload_label), label="Video source")
                source_note = gr.Markdown("Public footage without ground truth is qualitative only. MVTec/VisA models are not trained for this conveyor scene.")
                video = gr.Video(label="Existing video upload (MP4, AVI, MOV)", sources=["upload"])
                category = gr.Dropdown(choices=INSPECTION_CATEGORIES, value="bottle", label="Trained product category")
                model = gr.Dropdown(choices=["efficientad", "patchcore"], value="efficientad", label="Architecture")
                interval = gr.Slider(1, 120, value=config.get("video.sampling_interval", 15), step=1, label="Analyze every Nth frame")
                duration = gr.Slider(1, 60, value=5, step=1, label="Maximum seconds to process")
                threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Suspicious-frame threshold")
                consecutive = gr.Slider(1, 10, value=3, step=1, label="Minimum consecutive suspicious observations")
                gap = gr.Slider(0, 15, value=2, step=1, label="Merge gap (sample intervals)")
                min_duration = gr.Number(value=0, minimum=0, label="Minimum event duration (seconds)")
                roi = gr.Textbox(label="Optional fixed product ROI: x,y,width,height in source pixels", placeholder="Leave blank to inspect the full frame")
                annotations = gr.File(label="Optional existing ground-truth JSON (complete temporal intervals)", file_types=[".json"], type="filepath")
                run = gr.Button("Inspect Video", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Source, limits and evaluation status", interactive=False)
                metadata = gr.JSON(label="Video metadata and report")
                output = gr.Video(label="Annotated sampled frames; unsampled frames marked", interactive=False)
                timeline = gr.LinePlot(x="timestamp_s", y="anomaly_score", title="Sampled frame scores over time", x_title="Seconds", y_title="Calibrated anomaly score")
                gallery = gr.Gallery(label="Sampled frame heatmap overlays", columns=3)
                events = gr.Dataframe(label="Suspicious time intervals", interactive=False)
                frames_csv = gr.File(label="Frame scores CSV")
                events_csv = gr.File(label="Inspection events CSV")
                report_file = gr.File(label="JSON report with source attribution and evaluation status")

        def describe_source(choice):
            if choice not in sources:
                return "Upload an existing public or other permitted video. Without covered ground-truth annotations, results are qualitative."
            item = sources[choice][1]
            return f"**{item['sequence_type']}** — {item.get('license', '')}. {item.get('suitability', item.get('changes', ''))} " + str(item.get('source_page', ''))

        source.change(describe_source, source, source_note)

        def inspect(choice, uploaded, cat, architecture, sample, seconds, cutoff, count, merge, minimum, crop, gt_file):
            if choice in sources:
                path, provenance = sources[choice]
            else:
                path, provenance = uploaded, dict(sequence_type="uploaded_video", annotation_type="none")
            if not path:
                raise gr.Error("Select a prepared public example or upload an existing video.")
            try:
                region = tuple(int(value.strip()) for value in crop.split(',')) if crop.strip() else None
                truth = json.loads(Path(gt_file).read_text(encoding="utf-8")) if gt_file else None
                folder = Path(config.get("storage.artifacts_dir", "./artifacts")) / f"VID_{uuid.uuid4().hex[:12]}"
                pipeline = VideoInspectionPipeline(model_name=architecture, category=cat, sampling_interval=int(sample),
                    decision_threshold=cutoff, consecutive_frames=int(count), merge_gap_frames=int(merge),
                    min_event_duration_s=minimum, roi=region, max_analysis_frames=config.get("video.max_analysis_frames", 120))
                result = pipeline.process_video(path, folder / "annotated.mp4", max_duration_s=seconds,
                                                provenance=provenance, annotations=truth)
                meta, frame_df = result['metadata'], result['frame_dataframe']
                InspectionRepository().insert_inspection(folder.name, Path(path).name, "video", cat, architecture,
                    float(frame_df.anomaly_score.max()), "REVIEW_REQUIRED" if result['metrics']['evaluation_mode'] == 'qualitative' or len(result['event_dataframe']) else "LIKELY_NORMAL",
                    artifacts_path=str(folder))
                text = ("QUALITATIVE ONLY: no covered ground truth; no defect-detection accuracy reported."
                        if result['metrics']['evaluation_mode'] == 'qualitative' else "Metrics computed only against supplied covered ground truth.")
                if meta['truncated']:
                    text += " Processing was limited; see processed duration in metadata."
                text += " Events are time clusters, not tracked product counts."
                return (text, dict(metadata=meta, metrics=result['metrics']), str(folder / 'annotated.mp4'),
                        frame_df, [(g['overlay'], f"{g['timestamp_s']:.2f}s; {g['score']:.3f}") for g in result['suspicious_gallery']],
                        result['event_dataframe'], str(folder / 'frame_scores.csv'), str(folder / 'inspection_events.csv'),
                        str(folder / 'inspection_report.json'))
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        run.click(inspect, [source, video, category, model, interval, duration, threshold, consecutive, gap,
                           min_duration, roi, annotations],
                  [status, metadata, output, timeline, gallery, events, frames_csv, events_csv, report_file],
                  concurrency_id="model_inference", concurrency_limit=1)
