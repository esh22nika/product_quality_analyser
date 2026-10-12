from pathlib import Path
import tempfile
import gradio as gr
import pandas as pd

from visionguard.inference.video_inference import VideoInspectionPipeline

def create_video_tab():
    with gr.Tab("Video Inspection"):
        gr.Markdown("### Conveyor / Stream Video Inspection & Temporal Anomaly Event Aggregation")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Inspection Video (MP4/AVI)")
                category_dropdown = gr.Dropdown(
                    choices=["bottle", "cable", "capsule", "hazelnut", "metal_nut", "screw", "carpet", "leather"],
                    value="bottle",
                    label="Product Category"
                )
                model_dropdown = gr.Dropdown(
                    choices=["patchcore", "efficientad"],
                    value="patchcore",
                    label="Model Architecture"
                )
                sample_interval = gr.Slider(1, 10, value=2, step=1, label="Frame Sampling Interval")
                d_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Frame Anomaly Threshold")
                consec_frames = gr.Slider(1, 10, value=3, step=1, label="Min Consecutive Suspicious Frames for Event")
                merge_gap = gr.Slider(1, 15, value=5, step=1, label="Event Merging Gap (Frames)")

                inspect_video_btn = gr.Button("Inspect Video Stream", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    proc_time_box = gr.Number(label="Processing Time (s)", value=0.0, interactive=False)
                    throughput_box = gr.Number(label="Throughput (FPS)", value=0.0, interactive=False)
                    event_count_box = gr.Number(label="Aggregated Defect Events", value=0, interactive=False)

                annotated_video_out = gr.Video(label="Annotated Output Video", interactive=False)
                events_table = gr.Dataframe(label="Aggregated Defect Events Timeline", interactive=False)
                
                with gr.Row():
                    frame_csv_out = gr.File(label="Download Frame Scores CSV")
                    event_csv_out = gr.File(label="Download Events CSV")

        def run_video_inspection(v_path, cat, model_name, s_int, thresh, c_frames, m_gap):
            if v_path is None:
                empty_events = pd.DataFrame(columns=["event_id", "start_time_s", "end_time_s", "duration_s"])
                return 0.0, 0.0, 0, None, empty_events, None, None

            out_v_path = Path(tempfile.gettempdir()) / "visionguard_annotated.mp4"
            pipeline = VideoInspectionPipeline(
                model_name=model_name,
                category=cat,
                sampling_interval=int(s_int),
                decision_threshold=thresh,
                consecutive_frames=int(c_frames),
                merge_gap_frames=int(m_gap)
            )

            res = pipeline.process_video(
                video_path=v_path,
                output_video_path=out_v_path,
                max_duration_s=60.0
            )

            meta = res["metadata"]
            df_frames = res["frame_dataframe"]
            df_events = res["event_dataframe"]

            frame_csv = Path(tempfile.gettempdir()) / "frame_scores.csv"
            event_csv = Path(tempfile.gettempdir()) / "defect_events.csv"
            df_frames.to_csv(frame_csv, index=False)
            df_events.to_csv(event_csv, index=False)

            return (
                meta["processing_time_s"],
                meta["throughput_fps"],
                len(df_events),
                str(out_v_path) if out_v_path.exists() else None,
                df_events,
                str(frame_csv),
                str(event_csv)
            )

        inspect_video_btn.click(
            fn=run_video_inspection,
            inputs=[video_input, category_dropdown, model_dropdown, sample_interval, d_thresh, consec_frames, merge_gap],
            outputs=[proc_time_box, throughput_box, event_count_box, annotated_video_out, events_table, frame_csv_out, event_csv_out]
        )
