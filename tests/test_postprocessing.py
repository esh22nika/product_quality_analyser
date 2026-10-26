import numpy as np
from visionguard.inference.postprocessing import AnomalyPostprocessor

def test_heatmap_generation():
    # 100x100 anomaly map
    amap = np.linspace(0.0, 1.0, 10000).reshape((100, 100)).astype(np.float32)
    heatmap = AnomalyPostprocessor.generate_heatmap(amap)
    assert heatmap.shape == (100, 100, 3)
    assert heatmap.dtype == np.uint8


def test_relative_colors_reveal_narrow_range_without_mutating_scores():
    score_map = np.linspace(.497, .510, 10000).reshape(100, 100).astype(np.float32)
    original = score_map.copy()
    relative = AnomalyPostprocessor.generate_heatmap(score_map, scale="relative")
    fixed = AnomalyPostprocessor.generate_heatmap(score_map)
    assert np.unique(relative.reshape(-1, 3), axis=0).shape[0] > 100
    assert np.unique(fixed.reshape(-1, 3), axis=0).shape[0] < 10
    assert np.array_equal(score_map, original)


def test_constant_relative_map_does_not_invent_structure():
    colors = AnomalyPostprocessor.generate_heatmap(np.full((32, 32), .5, np.float32), scale="relative")
    assert np.unique(colors.reshape(-1, 3), axis=0).shape[0] == 1

def test_overlay_generation():
    orig = np.full((100, 100, 3), 128, dtype=np.uint8)
    heatmap = np.full((100, 100, 3), 255, dtype=np.uint8)
    overlay = AnomalyPostprocessor.generate_overlay(orig, heatmap, alpha=0.5)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8

def test_region_analysis():
    binary_mask = np.zeros((100, 100), dtype=np.uint8)
    binary_mask[20:40, 20:40] = 255  # 20x20 = 400 pixels
    amap = np.full((100, 100), 0.8, dtype=np.float32)

    res = AnomalyPostprocessor.analyze_regions(binary_mask, amap, min_region_area=50)
    assert res["num_regions"] == 1
    assert res["total_suspicious_pixels"] == 400
    assert res["estimated_anomalous_area_percent"] == 4.0
