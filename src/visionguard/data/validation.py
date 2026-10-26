"""Dataset integrity checks without modifying original benchmark files."""
from pathlib import Path
import csv

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def images(folder):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS] if folder.is_dir() else []

class DatasetValidator:
    SUPPORTED_DATASETS = ["mvtec_ad", "visa"]

    @staticmethod
    def validate_mvtec_ad(root, categories=None):
        root = Path(root)
        result = dict(valid=False, root=str(root), categories={}, warnings=[], total_categories=0)
        if not root.is_dir():
            result["error"] = f"Root directory '{root}' does not exist."
            return result
        available = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        targets = categories if categories is not None else available
        for category in targets:
            folder = root / category
            train = images(folder / "train/good")
            tests = {p.name: images(p) for p in (folder / "test").iterdir() if p.is_dir()} if (folder / "test").is_dir() else {}
            masks = {p.name: images(p) for p in (folder / "ground_truth").iterdir() if p.is_dir()} if (folder / "ground_truth").is_dir() else {}
            missing = []
            for kind, files in tests.items():
                if kind == "good":
                    continue
                for image in files:
                    candidates = [folder / "ground_truth" / kind / f"{image.stem}{suffix}.png" for suffix in ['_mask', '']]
                    if not any(p.exists() for p in candidates):
                        missing.append(str(image))
            valid = bool(train) and sum(map(len, tests.values())) > 0 and not missing
            result['categories'][category] = dict(valid=valid, train_good_count=len(train),
                test_counts={k:len(v) for k,v in tests.items()}, test_total=sum(map(len,tests.values())),
                ground_truth_counts={k:len(v) for k,v in masks.items()}, missing_masks=missing)
            if not valid:
                result['warnings'].append(f"Incomplete {category}: normal training={len(train)}, missing masks={len(missing)}")
        result['total_categories'] = len(result['categories'])
        result['valid'] = bool(targets) and all(item['valid'] for item in result['categories'].values())
        if not result['valid']:
            result['error'] = "No complete categories found, or one or more requested categories are incomplete."
        return result

    @staticmethod
    def validate_visa(root, categories=None):
        root = Path(root)
        if (root / 'visa_pytorch').is_dir():
            return DatasetValidator.validate_mvtec_ad(root / 'visa_pytorch', categories)
        csv_path = root / 'split_csv/1cls.csv'
        if not csv_path.exists():
            return dict(valid=False, error=f"Missing VisA split: {csv_path}", categories={}, warnings=[])
        result = dict(valid=True, root=str(root), categories={}, warnings=[])
        with csv_path.open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                category = row['object']
                if categories is not None and category not in categories:
                    continue
                info = result['categories'].setdefault(category, dict(valid=True, train_good_count=0, test_counts={}, test_total=0, ground_truth_counts={}))
                exists = (root / row['image']).is_file()
                if row['split'] == 'train':
                    info['train_good_count'] += int(row['label'] == 'normal' and exists)
                else:
                    info['test_total'] += 1
                    info['test_counts'][row['label']] = info['test_counts'].get(row['label'],0) + 1
                    if row['label'] != 'normal':
                        exists = exists and bool(row['mask']) and (root / row['mask']).is_file()
                info['valid'] = info['valid'] and exists
        if categories is not None and any(c not in result['categories'] for c in categories):
            result['valid'] = False
        result['total_categories'] = len(result['categories'])
        result['valid'] = result['valid'] and bool(result['categories']) and all(v['valid'] and v['train_good_count'] > 0 and v['test_total'] > 0 for v in result['categories'].values())
        if not result['valid']:
            result['error'] = 'Missing image/mask files or empty categories referenced by official VisA split.'
        return result
