import json
import zipfile
import numpy as np
import pytest
from visionguard.storage.export import export_image_result
from visionguard.storage.artifacts import ArtifactManager


def test_export_contains_real_arrays_and_report(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = dict(original_image=image, heatmap=image, annotated_overlay=image, thresholded_mask=image[:, :, 0],
                  anomaly_map=np.ones((8, 8), dtype=np.float32), image_score=.75, success=True)
    archive = export_image_result(result, tmp_path / 'INSP_test')
    with zipfile.ZipFile(archive) as z:
        assert 'anomaly_maps.npz' in z.namelist()
        assert json.loads(z.read('report.json'))['image_score'] == .75


def test_cleanup_preserves_noninspection_work_and_rejects_escape(tmp_path):
    (tmp_path / 'notes').mkdir()
    (tmp_path / 'INSP_a').mkdir()
    manager = ArtifactManager(tmp_path)
    with pytest.raises(ValueError):
        manager.cleanup_inspection('../outside')
    manager.cleanup_all()
    assert (tmp_path / 'notes').is_dir()
    assert not (tmp_path / 'INSP_a').exists()
