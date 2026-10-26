"""Run real image inference using an existing public benchmark sample."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='mvtec_anomaly_detection/bottle/test/good/000.png')
    parser.add_argument('--model', choices=['patchcore','efficientad'], default='efficientad')
    parser.add_argument('--category', default='bottle')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from visionguard.inference.image_inference import ImageInspectionPipeline
    from visionguard.storage.export import export_image_result
    result=ImageInspectionPipeline(args.model,args.category).inspect(args.image)
    if not result['success']:
        raise RuntimeError(result['error'])
    archive=export_image_result(result, Path('artifacts') / 'INSP_cli_demo')
    print(json.dumps({k:result[k] for k in ['image_score','predicted_label','inference_ms','checkpoint']},indent=2))
    print(archive)
