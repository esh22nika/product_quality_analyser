from pathlib import Path
import tempfile
import gradio as gr
import pandas as pd

from visionguard.storage.repository import InspectionRepository
from visionguard.data.categories import INSPECTION_CATEGORIES

def create_history_tab():
    repo = InspectionRepository()

    with gr.Tab("Inspection History"):
        gr.Markdown("### Local SQLite Inspection Audit Log & Human Feedback")

        with gr.Row():
            with gr.Column(scale=1):
                cat_filter = gr.Dropdown(
                    choices=["All"] + INSPECTION_CATEGORIES,
                    value="All",
                    label="Filter by Category"
                )
                status_filter = gr.Dropdown(
                    choices=["All", "LIKELY_NORMAL", "REVIEW_REQUIRED", "LIKELY_ANOMALOUS"],
                    value="All",
                    label="Filter by Decision"
                )
                refresh_history_btn = gr.Button("Refresh Log", variant="primary")
                export_csv_btn = gr.Button("Export Audit CSV")
                clear_history_btn = gr.Button("Clear History (Retention)", variant="stop")

            with gr.Column(scale=2):
                with gr.Row():
                    selected_id = gr.Textbox(label="Inspection ID for Feedback", placeholder="e.g. INSP_abc123")
                    feedback_choice = gr.Dropdown(
                        choices=["Confirmed Defect", "Normal Variation", "Uncertain", "Image Quality Issue"],
                        value="Confirmed Defect",
                        label="Inspector Decision"
                    )
                    notes_box = gr.Textbox(label="Inspector Notes", placeholder="Surface scratch near top rim")
                    submit_feedback_btn = gr.Button("Submit Human Review")

                feedback_status = gr.Textbox(label="Action Status", value="Ready", interactive=False)
                history_table = gr.Dataframe(label="Inspection History Records", interactive=False)
                download_audit_file = gr.File(label="Download Full Audit CSV")

        def load_history(cat, stat):
            df = repo.list_inspections(category=cat, decision_status=stat)
            return df

        def export_history():
            df = repo.list_inspections()
            tmp_p = Path(tempfile.gettempdir()) / "visionguard_audit_log.csv"
            df.to_csv(tmp_p, index=False)
            return str(tmp_p)

        def clear_records():
            repo.clear_all()
            empty_df = pd.DataFrame(columns=["id", "inspection_id", "timestamp", "category", "decision_status"])
            return "Cleared all records.", empty_df

        def submit_feedback(insp_id, choice, notes):
            if not insp_id:
                return "Error: Please specify an Inspection ID."
            repo.update_feedback(inspection_id=insp_id.strip(), label=choice, notes=notes)
            return f"Updated feedback for {insp_id} -> {choice}."

        refresh_history_btn.click(
            fn=load_history,
            inputs=[cat_filter, status_filter],
            outputs=[history_table]
        )

        export_csv_btn.click(
            fn=export_history,
            inputs=[],
            outputs=[download_audit_file]
        )

        clear_history_btn.click(
            fn=clear_records,
            inputs=[],
            outputs=[feedback_status, history_table]
        )

        submit_feedback_btn.click(
            fn=submit_feedback,
            inputs=[selected_id, feedback_choice, notes_box],
            outputs=[feedback_status]
        )
