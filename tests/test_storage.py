from pathlib import Path
from visionguard.storage.repository import InspectionRepository

def test_database_crud(tmp_path):
    db_file = tmp_path / "test.db"
    repo = InspectionRepository(db_path=db_file)

    repo.insert_inspection(
        inspection_id="INSP_001",
        input_filename="sample.png",
        input_type="image",
        category="bottle",
        model_name="patchcore",
        image_score=0.85,
        decision_status="LIKELY_ANOMALOUS"
    )

    df = repo.list_inspections()
    assert len(df) == 1
    assert df.iloc[0]["inspection_id"] == "INSP_001"
    assert df.iloc[0]["decision_status"] == "LIKELY_ANOMALOUS"

    repo.update_feedback("INSP_001", "Confirmed Defect", "Scratch visible")
    df_updated = repo.list_inspections()
    assert df_updated.iloc[0]["review_label"] == "Confirmed Defect"

    repo.clear_all()
    assert len(repo.list_inspections()) == 0
