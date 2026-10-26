import argparse
import json
from pathlib import Path
import sys

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionguard.data.validation import DatasetValidator

def main():
    parser = argparse.ArgumentParser(description="Validate local industrial dataset structure.")
    parser.add_argument("--dataset", type=str, default="mvtec_ad", help="Dataset name (mvtec_ad, visa)")
    parser.add_argument("--root", type=str, default="./datasets/mvtec_ad", help="Root dataset directory")
    args = parser.parse_args()

    print(f"Validating dataset '{args.dataset}' at path: {args.root}...")
    res = (DatasetValidator.validate_visa(args.root) if args.dataset == "visa" else DatasetValidator.validate_mvtec_ad(args.root))

    print("\n--- Validation Results ---")
    print(f"Dataset root exists: {res['valid']}")
    print(f"Total categories identified: {res.get('total_categories', 0)}")

    for cat, info in res.get("categories", {}).items():
        status = "READY" if info["valid"] else "INCOMPLETE"
        print(f"  • [{status}] {cat:<12}: Train Normal={info['train_good_count']}, Test Total={info['test_total']}")

    if res.get("warnings"):
        print("\nWarnings:")
        for w in res["warnings"]:
            print(f"  ⚠️  {w}")

    if not res["valid"]:
        print(f"\nError: {res.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
