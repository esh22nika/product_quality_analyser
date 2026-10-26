from pathlib import Path
import tempfile
import gradio as gr
import pandas as pd

from visionguard.inference.batch_inference import BatchInspectionPipeline
from visionguard.data.categories import INSPECTION_CATEGORIES

def create_batch_tab():
    with gr.Tab("Batch Inspection"):
        gr.Markdown("### Multi-Image High-Throughput Quality Control")

        with gr.Row():
            with gr.Column(scale=1):
                files_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Image Batch")
                category_dropdown = gr.Dropdown(
                    choices=INSPECTION_CATEGORIES,
                    value="bottle",
                    label="Product Category"
                )
                model_dropdown = gr.Dropdown(
                    choices=["patchcore", "efficientad"],
                    value="patchcore",
                    label="Anomaly Architecture"
                )
                decision_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Decision Threshold")
                
                run_batch_btn = gr.Button("Run Batch Inspection", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    total_count = gr.Number(label="Total Images", value=0, interactive=False)
                    normal_count = gr.Number(label="Normal", value=0, interactive=False)
                    review_count = gr.Number(label="Review Required", value=0, interactive=False)
                    anomaly_count = gr.Number(label="Anomalous", value=0, interactive=False)

                results_table = gr.Dataframe(label="Batch Inspection Results", interactive=False)
                download_csv = gr.File(label="Download CSV Report")

        def run_batch(files, cat, model_name, d_thresh):
            if not files:
                empty_df = pd.DataFrame(columns=["filename", "status", "predicted_label", "image_score"])
                return 0, 0, 0, 0, empty_df, None

            pipeline = BatchInspectionPipeline(model_name=model_name, category=cat)
            file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
            res = pipeline.inspect_batch(file_paths, decision_threshold=d_thresh)

            df = res["table"]
            summary = res["summary"]

            # Save temporary CSV for download
            tmp_csv = Path(tempfile.gettempdir()) / "visionguard_batch_report.csv"
            df.to_csv(tmp_csv, index=False)

            return (
                summary["total_images"],
                summary["normal_count"],
                summary["review_count"],
                summary["anomalous_count"],
                df,
                str(tmp_csv)
            )

        run_batch_btn.click(
            concurrency_id="model_inference", concurrency_limit=1,
            fn=run_batch,
            inputs=[files_input, category_dropdown, model_dropdown, decision_thresh],
            outputs=[total_count, normal_count, review_count, anomaly_count, results_table, download_csv]
        )
