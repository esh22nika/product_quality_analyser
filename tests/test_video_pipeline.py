import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
from visionguard.inference.video_inference import VideoInspectionPipeline


class Detector:
    threshold = .5
    device = 'cpu'
    checkpoint_path = 'test-fixture'
    def predict(self, image):
        value = float(image.mean() > 100)
        return dict(image_score=value, raw_score=value, anomaly_map=np.full(image.shape[:2], value, np.float32),
                    predicted_label='LIKELY_ANOMALOUS' if value else 'LIKELY_NORMAL', error=None)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip('MJPG writer unavailable')
    for i in range(20):
        writer.write(np.full((48, 64, 3), 200 if 5 <= i < 15 else 0, np.uint8))
    writer.release()
    return path


def test_video_outputs_heatmaps_events_and_no_fabricated_metrics(clip, tmp_path, monkeypatch):
    monkeypatch.setattr('visionguard.inference.video_inference.ModelManager.get_model', lambda *a, **k: Detector())
    pipeline = VideoInspectionPipeline(sampling_interval=1, consecutive_frames=3)
    output = tmp_path / 'run/annotated.mp4'
    progress = []
    result = pipeline.process_video(clip, output, on_progress=lambda fraction, message: progress.append((fraction, message)))
    assert progress[0][0] == 0 and progress[-1][0] == 1
    assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))
    assert sum('sampled frames ·' in message for _, message in progress) == 20
    assert 'relative' in result['metadata']['heatmap_visualization']
    assert result['metadata']['processed_frames'] == 20
    assert len(result['event_dataframe']) == 1
    assert result['metrics']['frame_auroc'] is None
    assert len(list((output.parent / 'maps').glob('*.npz'))) == 20
    assert (output.parent / 'inspection_report.json').exists()
    saved = json.loads((output.parent / 'inspection_report.json').read_text())
    assert saved['metrics']['evaluation_mode'] == 'qualitative'
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    cap.release()


def test_labeled_video_and_truncation_coverage(clip, tmp_path, monkeypatch):
    monkeypatch.setattr('visionguard.inference.video_inference.ModelManager.get_model', lambda *a, **k: Detector())
    truth = dict(annotation_type='temporal_intervals', complete=True, coverage_end_s=2,
                 events=[dict(start_time_s=.5, end_time_s=1.5)])
    result = VideoInspectionPipeline(sampling_interval=1).process_video(clip, annotations=truth)
    assert result['metrics']['frame_auroc'] == 1
    assert result['metrics']['event_precision'] == 1
    truth['coverage_end_s'] = 1
    with pytest.raises(ValueError, match='cover'):
        VideoInspectionPipeline(sampling_interval=1).process_video(clip, annotations=truth)


def test_separated_single_positives_do_not_form_consecutive_event():
    df = pd.DataFrame([dict(frame_idx=i, anomaly_score=float(i % 2 == 0), is_anomalous=i % 2 == 0) for i in range(8)])
    assert VideoInspectionPipeline(sampling_interval=1, consecutive_frames=2)._aggregate_events(df, 10) == []


def test_bad_roi_and_inference_failure_are_errors(clip, monkeypatch):
    monkeypatch.setattr('visionguard.inference.video_inference.ModelManager.get_model', lambda *a, **k: Detector())
    with pytest.raises(ValueError, match='ROI'):
        VideoInspectionPipeline(roi=(100, 100, 4, 4)).process_video(clip)
    detector = Detector()
    detector.predict = lambda image: dict(error='checkpoint failed', anomaly_map=None)
    monkeypatch.setattr('visionguard.inference.video_inference.ModelManager.get_model', lambda *a, **k: detector)
    with pytest.raises(RuntimeError, match='checkpoint failed'):
        VideoInspectionPipeline().process_video(clip)


def test_synthetic_label_banner_is_excluded_from_product_roi(clip, monkeypatch):
    seen = []
    detector = Detector()
    original = detector.predict
    def capture(image):
        seen.append(image.shape[:2])
        return original(image)
    detector.predict = capture
    monkeypatch.setattr('visionguard.inference.video_inference.ModelManager.get_model', lambda *a, **k: detector)
    result = VideoInspectionPipeline().process_video(clip, provenance=dict(sequence_type='synthetic_image_sequence', recommended_roi=[0, 8, 64, 40]))
    assert seen and all(shape == (40, 64) for shape in seen)
    assert result['metrics']['frame_auroc'] is None
