import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionguard.models.loader import ModelManager
from visionguard.evaluation.image_metrics import ImageMetricsCalculator
from visionguard.evaluation.pixel_metrics import PixelMetricsCalculator
from visionguard.evaluation.reports import BenchmarkReporter

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model on test benchmark annotations.")
    parser.add_argument("--model", type=str, required=True, choices=["patchcore", "efficientad"])
    parser.add_argument("--dataset", type=str, default="mvtec_ad")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--root", type=str, default="./datasets/mvtec_ad")
    parser.add_argument("--reports", type=str, default="./reports")
    args = parser.parse_args()

    cat_dir = Path(args.root) / args.category
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}. Please place dataset.")
        sys.exit(1)

    print(f"Evaluating {args.model.upper()} on {args.category}...")
    model = ModelManager.get_model(model_name=args.model, category=args.category)

    y_true = []
    y_scores = []
    gt_masks = []
    anomaly_maps = []

    # Iterate through test categories
    for sub in test_dir.iterdir():
        if not sub.is_dir():
            continue
        is_normal = (sub.name == "good")
        for img_p in sub.glob("*.png"):
            img = cv2.imread(str(img_p))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = model.predict(img_rgb)

            label = 0 if is_normal else 1
            y_true.append(label)
            y_scores.append(pred["image_score"])

            # Check for ground truth mask
            if not is_normal and gt_dir.exists():
                gt_mask_p = gt_dir / sub.name / f"{img_p.stem}_mask.png"
                if gt_mask_p.exists():
                    mask = cv2.imread(str(gt_mask_p), cv2.IMREAD_GRAYSCALE)
                    gt_masks.append(mask)
                    anomaly_maps.append(pred["anomaly_map"])

    img_metrics = ImageMetricsCalculator.calculate(y_true, y_scores)
    print("\n--- Image-Level Benchmark Metrics ---")
    print(f"AUROC: {img_metrics.get('auroc')}")
    print(f"Average Precision: {img_metrics.get('average_precision')}")
    print(f"F1 Score: {img_metrics.get('f1_score')}")

    pixel_metrics = {}
    if gt_masks:
        pixel_metrics = PixelMetricsCalculator.calculate(gt_masks, anomaly_maps)
        print("\n--- Pixel-Level Localization Metrics ---")
        print(f"Pixel AUROC: {pixel_metrics.get('pixel_auroc')}")
        print(f"Dice: {pixel_metrics.get('dice')}")
        print(f"IoU: {pixel_metrics.get('iou')}")

    # Save to report
    reporter = BenchmarkReporter(args.reports)
    exp_id = f"eval_{args.model}_{args.category}"
    combined_metrics = {**img_metrics, **pixel_metrics}
    json_path = reporter.save_experiment_report(
        experiment_id=exp_id,
        metrics=combined_metrics,
        model_info={"name": args.model},
        dataset_info={"name": args.dataset, "category": args.category}
    )
    print(f"\nReport saved to: {json_path}")

if __name__ == "__main__":
    main()
