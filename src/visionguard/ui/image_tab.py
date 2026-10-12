import uuid
from pathlib import Path
import gradio as gr
import numpy as np

from visionguard.inference.image_inference import ImageInspectionPipeline
from visionguard.storage.repository import InspectionRepository
from visionguard.storage.artifacts import ArtifactManager

def create_image_tab():
    repo = InspectionRepository()
    artifact_mgr = ArtifactManager()

    with gr.Tab("Image Inspection"):
        gr.Markdown("### Single Product Image Inspection & Anomaly Heatmap Localization")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Upload Product Image")
                category_dropdown = gr.Dropdown(
                    choices=["bottle", "cable", "capsule", "hazelnut", "metal_nut", "screw", "carpet", "leather"],
                    value="bottle",
                    label="Product Category"
                )
                model_dropdown = gr.Dropdown(
                    choices=["patchcore", "efficientad"],
                    value="patchcore",
                    label="Anomaly Architecture"
                )
                decision_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Decision Threshold")
                pixel_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Pixel Heatmap Threshold")
                min_region_area = gr.Slider(10, 500, value=50, step=10, label="Min Suspicious Region Area (px)")
                
                inspect_btn = gr.Button("Inspect Image", variant="primary")
                clear_btn = gr.Button("Clear")

            with gr.Column(scale=2):
                with gr.Row():
                    status_badge = gr.Textbox(label="Operational Decision", value="Awaiting Input", interactive=False)
                    score_box = gr.Number(label="Anomaly Score", value=0.0, interactive=False)
                    latency_box = gr.Number(label="Inference Latency (ms)", value=0.0, interactive=False)
                    area_box = gr.Number(label="Est. Anomalous Area (%)", value=0.0, interactive=False)

                warnings_box = gr.Textbox(label="Image Quality & System Notes", value="None", interactive=False)

                with gr.Row():
                    orig_display = gr.Image(label="Original Image", interactive=False)
                    heatmap_display = gr.Image(label="Anomaly Heatmap", interactive=False)

                with gr.Row():
                    overlay_display = gr.Image(label="Overlay with Bounding Boxes", interactive=False)
                    mask_display = gr.Image(label="Thresholded Suspicious Mask", interactive=False)

        def run_inspection(img, cat, model_name, d_thresh, p_thresh, min_area):
            if img is None:
                return (
                    "No Image Provided", 0.0, 0.0, 0.0,
                    "Please upload a valid product image.",
                    None, None, None, None
                )

            pipeline = ImageInspectionPipeline(model_name=model_name, category=cat)
            res = pipeline.inspect(
                img,
                decision_threshold=d_thresh,
                pixel_threshold=p_thresh,
                min_region_area=int(min_area)
            )

            if not res["success"]:
                err = res.get("error", "Unknown error")
                return (
                    "ERROR", 0.0, 0.0, 0.0,
                    f"Inference failed: {err}",
                    img, None, None, None
                )

            insp_id = f"INSP_{uuid.uuid4().hex[:8]}"
            artifact_mgr.save_image_artifact(insp_id, "overlay.png", res["annotated_overlay"])
            repo.insert_inspection(
                inspection_id=insp_id,
                input_filename=f"{cat}_sample.png",
                input_type="image",
                category=cat,
                model_name=model_name,
                image_score=res["image_score"],
                decision_status=res["predicted_label"],
                artifacts_path=str(artifact_mgr.artifacts_dir / insp_id)
            )

            warnings = res["quality"]["warnings"]
            warnings_text = "; ".join(warnings) if warnings else "Image quality checks passed."

            return (
                res["predicted_label"],
                res["image_score"],
                res["inference_ms"],
                res["region_analysis"]["estimated_anomalous_area_percent"],
                warnings_text,
                res["original_image"],
                res["heatmap"],
                res["annotated_overlay"],
                res["thresholded_mask"]
            )

        inspect_btn.click(
            fn=run_inspection,
            inputs=[image_input, category_dropdown, model_dropdown, decision_thresh, pixel_thresh, min_region_area],
            outputs=[
                status_badge, score_box, latency_box, area_box,
                warnings_box, orig_display, heatmap_display, overlay_display, mask_display
            ]
        )

        clear_btn.click(
            fn=lambda: (None, "Awaiting Input", 0.0, 0.0, 0.0, "None", None, None, None, None),
            inputs=[],
            outputs=[
                image_input, status_badge, score_box, latency_box, area_box,
                warnings_box, orig_display, heatmap_display, overlay_display, mask_display
            ]
        )
