"""Evaluate both architectures on exactly the same eligible image samples."""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

def main(single=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='mvtec_ad', choices=['mvtec_ad', 'visa'])
    parser.add_argument('--categories', nargs='+', default=['bottle'])
    parser.add_argument('--category', help='Single category alias')
    parser.add_argument('--models', nargs='+', default=['patchcore', 'efficientad'], choices=['patchcore', 'efficientad'])
    parser.add_argument('--model', choices=['patchcore', 'efficientad'], help='Single model alias')
    parser.add_argument('--root', default='./mvtec_anomaly_detection')
    parser.add_argument('--reports', default='./reports')
    parser.add_argument('--limit-per-class', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--corruptions', nargs='+', default=['none'], choices=['none', 'brightness', 'contrast', 'blur', 'jpeg'])
    args = parser.parse_args()
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error('--limit-per-class must be positive')
    import torch
    torch.set_num_threads(args.cpu_threads)
    from visionguard.evaluation.benchmark import evaluate
    from visionguard.evaluation.reports import BenchmarkReporter
    rows = []
    for category in [args.category] if args.category else args.categories:
        for name in [args.model] if args.model else args.models:
            baseline = None
            for corruption in args.corruptions:
                path, metrics = evaluate(name, args.dataset, category, args.root, args.reports, args.limit_per_class, args.threshold, corruption)
                row = dict(model=name, category=category, corruption=corruption, report=str(path), **metrics)
                if corruption == 'none':
                    baseline = metrics
                if baseline:
                    row['auroc_change_from_clean'] = metrics['auroc'] - baseline['auroc']
                rows.append(row)
                print(name, category, corruption, metrics, flush=True)
                gc.collect()
    print(BenchmarkReporter(args.reports).save_comparison_table(rows, f'verified_comparison_{args.dataset}.csv'))

if __name__ == '__main__':
    main()
