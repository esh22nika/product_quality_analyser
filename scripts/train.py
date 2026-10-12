import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionguard.models.registry import ModelRegistry
from visionguard.data.validation import DatasetValidator

def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model on normal product images.")
    parser.add_argument("--model", type=str, required=True, choices=["patchcore", "efficientad"], help="Model architecture")
    parser.add_argument("--dataset", type=str, default="mvtec_ad", help="Benchmark dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Product category")
    parser.add_argument("--root", type=str, default="./datasets/mvtec_ad", help="Dataset root path")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints", help="Checkpoints directory")
    args = parser.parse_args()

    # 1. Validate dataset existence
    val_res = DatasetValidator.validate_mvtec_ad(args.root, categories=[args.category])
    cat_info = val_res.get("categories", {}).get(args.category)
    if not cat_info or cat_info.get("train_good_count", 0) == 0:
        print(f"Error: Dataset for category '{args.category}' is missing or empty at {args.root}/{args.category}/train/good.")
        print("Please download MVTec AD and place the category files properly before training.")
        sys.exit(1)

    print(f"Starting training: Model={args.model.upper()}, Category={args.category}, Samples={cat_info['train_good_count']}")
    
    # 2. Get registered model
    model_cls = ModelRegistry.get(args.model)
    model = model_cls()

    config = {
        "root": args.root,
        "category": args.category,
        "checkpoint_dir": args.checkpoints
    }

    result = model.train(config)
    print("\n--- Training Finished Successfully ---")
    print(f"Model: {result['model_name']}")
    print(f"Category: {result['category']}")
    print(f"Duration: {result['training_time_s']:.2f}s")
    print(f"Checkpoint saved at: {result['checkpoint_path']}")

if __name__ == "__main__":
    main()
