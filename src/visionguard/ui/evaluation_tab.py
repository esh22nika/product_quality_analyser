from pathlib import Path
import gradio as gr
import pandas as pd
import json
import plotly.graph_objects as go
from visionguard.data.categories import INSPECTION_CATEGORIES

def create_evaluation_tab():
    with gr.Tab("Model Evaluation"):
        gr.Markdown("### Quantitative Evaluation & Benchmark Metrics (MVTec AD / VisA)")

        with gr.Row():
            with gr.Column(scale=1):
                eval_model = gr.Dropdown(choices=["Both", "PatchCore", "EfficientAD"], value="Both", label="Model Architecture")
                eval_category = gr.Dropdown(choices=INSPECTION_CATEGORIES + ["all"], value="bottle", label="Benchmark Category")
                load_reports_btn = gr.Button("Load Evaluation Reports", variant="primary")

            with gr.Column(scale=2):
                eval_summary = gr.Textbox(label="Evaluation Status", value="Select parameters and load metrics.", interactive=False)
                metrics_table = gr.Dataframe(label="Image & Pixel Level Metrics", interactive=False)
                comparison_plot = gr.Plot(label="Measured AUROC by category, subset size and corruption")
                robustness_notes = gr.Markdown("""
                > [!TIP]
                > **Reproducibility Guarantee:**
                > Metrics shown reflect runs against ground-truth defect annotations (AUROC, AP, F1, Pixel AUROC, IoU).
                > No synthetic or fabricated metrics are reported. Run `python scripts/evaluate.py` to generate reports.
                """)

        def load_reports(m_name, cat):
            reports_dir = Path("./reports")
            if not reports_dir.exists():
                return "Reports directory './reports' does not exist yet. Run evaluation scripts first.", pd.DataFrame(), None

            records = []
            for json_file in reports_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("schema_version") != 2:
                            continue  # Legacy pixel IoU/Dice were incorrect.
                        m_info = data.get("model_info", {})
                        metrics = data.get("metrics", {})
                        if (m_name == "Both" or m_name.lower() in m_info.get("name", "").lower()) and (cat == "all" or data.get("dataset_info", {}).get("category") == cat):
                            records.append({
                                "Protocol": data.get("dataset_info", {}).get("split", "unknown"),
                                "Samples": metrics.get("sample_count"),
                                "Corruption": data.get("dataset_info", {}).get("corruption", "none"),
                                "Report ID": data.get("experiment_id", json_file.stem),
                                "Model": m_info.get("name"),
                                "Category": data.get("dataset_info", {}).get("category"),
                                "Image AUROC": metrics.get("auroc", "N/A"),
                                "Average Precision": metrics.get("average_precision", "N/A"),
                                "F1 Score": metrics.get("f1_score", "N/A"),
                                "Pixel AUROC": metrics.get("pixel_auroc", "N/A"),
                                "IoU": metrics.get("iou", "N/A")
                            })
                except Exception:
                    continue

            if not records:
                return f"No corrected v2 evaluation reports found for {m_name}/{cat}; legacy reports are excluded.", pd.DataFrame(columns=["Report ID", "Model", "Category", "Image AUROC", "F1 Score"]), None

            df = pd.DataFrame(records)
            figure = go.Figure()
            for name in df["Model"].unique():
                subset = df[df["Model"] == name]
                figure.add_bar(name=name, x=[f"{r['Category']} / n={r['Samples']} / {r['Corruption']}" for _, r in subset.iterrows()], y=subset["Image AUROC"].tolist())
            figure.update_layout(barmode="group", yaxis_title="Image AUROC", yaxis_range=[0, 1])
            return f"Loaded {len(records)} corrected evaluation records. Check subset/provenance before comparing; publisher scores are not used.", df, figure

        load_reports_btn.click(
            fn=load_reports,
            inputs=[eval_model, eval_category],
            outputs=[eval_summary, metrics_table, comparison_plot]
        )
