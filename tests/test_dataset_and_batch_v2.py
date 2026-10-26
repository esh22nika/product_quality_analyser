from pathlib import Path
import numpy as np
from PIL import Image
from visionguard.data.validation import DatasetValidator
from visionguard.inference.batch_inference import BatchInspectionPipeline


def test_missing_mask_fails_validation(tmp_path):
    for folder in ['bottle/train/good', 'bottle/test/scratch']:
        path = tmp_path / folder
        path.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path / '000.png')
    report = DatasetValidator.validate_mvtec_ad(tmp_path)
    assert not report['valid']
    assert len(report['categories']['bottle']['missing_masks']) == 1


def test_batch_corrupt_file_does_not_cancel_others(tmp_path):
    batch = BatchInspectionPipeline()
    batch.pipeline.inspect = lambda path, **kwargs: dict(success=False, error='corrupt')
    result = batch.inspect_batch([tmp_path/'bad.png', tmp_path/'bad2.png'])
    assert result['summary']['failed_count'] == 2
    assert result['summary']['avg_inference_ms'] == 0
    assert batch.inspect_batch([])['summary']['total_images'] == 0
