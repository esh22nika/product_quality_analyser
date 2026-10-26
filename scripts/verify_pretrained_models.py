"""Exercise local trained weights on benchmark images, without training or remote code."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from visionguard.models.loader import ModelManager
from visionguard.data.categories import MVTEC_CATEGORIES
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['efficientad', 'patchcore'])
    parser.add_argument('--categories', nargs='+', default=MVTEC_CATEGORIES + ['pcb1'])
    parser.add_argument('--wait-for-downloads', type=int, default=0,
                        help='Maximum seconds to wait for each EfficientAD checkpoint.')
    args = parser.parse_args()
    results = []
    for model_name in args.models:
        for category in args.categories:
            checkpoint = ROOT / 'checkpoints/public_efficientad' / category / 'model.ckpt'
            if model_name == 'efficientad' and args.wait_for_downloads:
                deadline = time.monotonic() + args.wait_for_downloads
                while not checkpoint.is_file() and time.monotonic() < deadline:
                    time.sleep(2)
            image_root = (ROOT / 'datasets/visa/visa_pytorch' if category == 'pcb1'
                          else ROOT / 'mvtec_anomaly_detection') / category / 'test/good'
            image = next(iter(sorted(image_root.glob('*.*'))), None)
            if image is None:
                raise FileNotFoundError(image_root)
            model = ModelManager.get_model(model_name, category, checkpoint_dir=str(ROOT / 'checkpoints'), device='cpu')
            prediction = model.predict(str(image))
            if prediction.get('error'):
                raise RuntimeError(f'{model_name}/{category}: {prediction["error"]}')
            assert model.is_loaded
            assert np.isfinite(prediction['image_score'])
            assert prediction['anomaly_map'].ndim == 2
            assert np.isfinite(prediction['anomaly_map']).all()
            result = dict(model=model_name, category=category, image=str(image),
                          score=prediction['image_score'], seconds=prediction['inference_ms'] / 1000)
            results.append(result)
            print(json.dumps(result), flush=True)
            ModelManager.clear_cache()
            del model, prediction
            gc.collect()
    report = ROOT / 'artifacts/pretrained_model_verification.json'
    report.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'PASS: {len(results)} trained model/category inference checks. Report: {report}', flush=True)


if __name__ == '__main__':
    main()
