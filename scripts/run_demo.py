import os
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionguard.inference.image_inference import ImageInspectionPipeline
from visionguard.inference.video_inference import VideoInspectionPipeline

def main():
    print("Running VisionGuard end-to-end demo...")
    # Create synthetic test fixture for quick verification
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[50:200, 50:200] = [180, 180, 180]
    # Inject a surface defect
    test_img[90:110, 90:120] = [30, 30, 220]

    pipeline = ImageInspectionPipeline(model_name="patchcore", category="bottle")
    res = pipeline.inspect(test_img)

    print(f"Status: {res['success']}")
    print(f"Model: {res['model_name']}")
    print(f"Score: {res['image_score']}")
    print(f"Decision: {res['predicted_label']}")
    print(f"Inference latency: {res['inference_ms']} ms")
    print(f"Estimated suspicious area: {res['region_analysis']['estimated_anomalous_area_percent']}%")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
