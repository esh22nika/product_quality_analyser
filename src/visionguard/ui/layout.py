import gradio as gr
import torch

from visionguard.ui.image_tab import create_image_tab
from visionguard.ui.batch_tab import create_batch_tab
from visionguard.ui.video_tab import create_video_tab
from visionguard.ui.dataset_tab import create_dataset_tab
from visionguard.ui.evaluation_tab import create_evaluation_tab
from visionguard.ui.history_tab import create_history_tab

def create_app() -> gr.Blocks:
    """Builds the complete VisionGuard Gradio Blocks application."""
    device_name = "CUDA (GPU Acceleration)" if torch.cuda.is_available() else "CPU (Standard Mode)"

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    )

    with gr.Blocks(theme=custom_theme, title="VisionGuard — Industrial Anomaly Inspection") as demo:
        # Header
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"""
                # 🛡️ VisionGuard
                ### Robust Image and Video-Based Industrial Defect Detection
                *Anomaly localization, temporal video stream inspection, and benchmark verification.*
                """)
            with gr.Column(scale=1):
                gr.Markdown(f"""
                **Hardware Status:** `{device_name}`  
                **Engine:** `Anomalib + PyTorch`  
                **Active Models:** `PatchCore & EfficientAD`
                """)

        # Tabs
        create_image_tab()
        create_batch_tab()
        create_video_tab()
        create_dataset_tab()
        create_evaluation_tab()
        create_history_tab()

    return demo
