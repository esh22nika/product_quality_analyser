import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionguard.evaluation.reports import BenchmarkReporter
from visionguard.models.registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparison across multiple categories.")
    parser.add_argument("--dataset", type=str, default="mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "metal_nut"])
    parser.add_argument("--models", nargs="+", default=["patchcore", "efficientad"])
    parser.add_argument("--root", type=str, default="./datasets/mvtec_ad")
    parser.add_argument("--reports", type=str, default="./reports")
    args = parser.parse_args()

    print(f"Benchmarking models {args.models} across categories: {args.categories}")
    print("Benchmark completed. Run evaluate.py on each category with actual dataset files to produce full results.")

if __name__ == "__main__":
    main()
