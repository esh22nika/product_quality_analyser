import numpy as np
from visionguard.evaluation.image_metrics import ImageMetricsCalculator

def test_image_metrics_calculation():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    res = ImageMetricsCalculator.calculate(y_true, y_scores, threshold=0.5)
    
    assert res["auroc"] == 1.0
    assert res["average_precision"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["sample_count"] == 4

def test_single_class_metrics():
    y_true = [0, 0, 0]
    y_scores = [0.1, 0.2, 0.3]
    res = ImageMetricsCalculator.calculate(y_true, y_scores)
    assert res.get("auroc") is None
    assert "error" in res
