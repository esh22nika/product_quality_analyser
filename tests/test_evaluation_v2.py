import numpy as np
import pytest
from visionguard.evaluation.pixel_metrics import PixelMetricsCalculator
from visionguard.evaluation.image_metrics import ImageMetricsCalculator
from visionguard.evaluation.video_metrics import VideoMetricsCalculator
from visionguard.evaluation.region_metrics import aupro


def test_wrong_pixel_predictions_do_not_score_perfect_iou():
    mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    result = PixelMetricsCalculator.calculate([mask], [1 - mask.astype(float)])
    assert result['iou'] == 0 and result['dice'] == 0
    assert result['pixel_auroc'] == 0


def test_region_overlap_perfect_and_reversed():
    mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert aupro([mask], [mask.astype(float)]) == pytest.approx(1)
    assert aupro([mask], [1-mask.astype(float)]) == pytest.approx(0)


def test_calibration_decisions_are_separate_from_rank_scores():
    result = ImageMetricsCalculator.calculate([0, 1], [100., 200.], decision_scores=[.1, .9])
    assert result['f1_score'] == 1
    assert result['auroc'] == 1


def test_unlabeled_video_never_gets_accuracy_metrics():
    result = VideoMetricsCalculator.calculate()
    assert result['evaluation_mode'] == 'qualitative'
    assert result['event_precision'] is None


def test_one_prediction_cannot_match_two_truth_events():
    events = [dict(start_time_s=0, end_time_s=10)]
    truth = [dict(start_time_s=1, end_time_s=3), dict(start_time_s=6, end_time_s=8)]
    result = VideoMetricsCalculator.calculate(detected_events=events, ground_truth_events=truth)
    assert result['event_precision'] == 1
    assert result['event_recall'] == .5


def test_false_alert_is_not_a_duplicate():
    predicted = [dict(start_time_s=0, end_time_s=2), dict(start_time_s=8, end_time_s=9)]
    result = VideoMetricsCalculator.calculate(detected_events=predicted, ground_truth_events=[dict(start_time_s=0, end_time_s=2)])
    assert result['duplicate_alert_rate'] == 0
    assert result['event_precision'] == .5
