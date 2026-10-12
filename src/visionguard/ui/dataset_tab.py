from pathlib import Path
import gradio as gr
import pandas as pd

from visionguard.data.validation import DatasetValidator

def create_dataset_tab():
    with gr.Tab("Dataset Explorer"):
        gr.Markdown("### Industrial Benchmark Datasets Status & Split Verification")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_name = gr.Dropdown(choices=["mvtec_ad", "visa"], value="mvtec_ad", label="Dataset")
                dataset_root = gr.Textbox(value="./datasets/mvtec_ad", label="Dataset Path")
                refresh_btn = gr.Button("Scan & Validate Dataset", variant="primary")

            with gr.Column(scale=2):
                status_summary = gr.Textbox(label="Validation Summary", value="Click 'Scan' to verify local datasets.", interactive=False)
                categories_table = gr.Dataframe(label="Discovered Categories & Sample Counts", interactive=False)
                license_note = gr.Markdown("""
                > [!NOTE]
                > **Dataset Licence Restrictions:**
                > - **MVTec AD**: Strictly for non-commercial research & education under its published licence.
                > - **VisA**: Research / non-commercial use terms apply.
                > Please respect benchmark terms and place the downloaded subsets into `./datasets/mvtec_ad`.
                """)

        def scan_dataset(ds_name, root):
            res = DatasetValidator.validate_mvtec_ad(root)
            if not res["valid"]:
                return (
                    f"Dataset Invalid or Not Found: {res.get('error', 'Unknown issue')}",
                    pd.DataFrame(columns=["Category", "Train (Normal)", "Test (Total)", "Ground Truth Masks"])
                )

            rows = []
            for cat, data in res["categories"].items():
                rows.append({
                    "Category": cat,
                    "Train (Normal)": data["train_good_count"],
                    "Test (Total)": data["test_total"],
                    "Ground Truth Masks": sum(data["ground_truth_counts"].values()),
                    "Status": "Ready" if data["valid"] else "Incomplete"
                })

            df = pd.DataFrame(rows)
            summary = f"Discovered {len(rows)} categories under '{root}'. Found {df['Train (Normal)'].sum()} training images."
            return summary, df

        refresh_btn.click(
            fn=scan_dataset,
            inputs=[dataset_name, dataset_root],
            outputs=[status_summary, categories_table]
        )
