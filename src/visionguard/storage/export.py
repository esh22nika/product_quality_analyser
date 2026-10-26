"""Inspection packages containing real model outputs and a machine-readable report."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np


def export_image_result(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key in ['original_image', 'heatmap', 'annotated_overlay', 'thresholded_mask']:
        image = result[key]
        encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        if not cv2.imwrite(str(folder / f'{key}.png'), encoded):
            raise OSError(f'Could not save {key}')
    np.savez_compressed(folder / 'anomaly_maps.npz', calibrated=result['anomaly_map'],
                        raw=result.get('raw_anomaly_map', result['anomaly_map']))
    report = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return _archive(folder)


def _archive(folder):
    # Place the ZIP next to the folder so it cannot include itself.
    return shutil.make_archive(str(folder), 'zip', root_dir=folder)
