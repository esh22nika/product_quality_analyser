"""Inspect existing footage and export a video, score/event CSVs and provenance report."""
import argparse
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--model', choices=['patchcore', 'efficientad'], default='efficientad')
    parser.add_argument('--category', default='bottle')
    parser.add_argument('--sampling-interval', type=int, default=15)
    parser.add_argument('--max-duration', type=float, default=5)
    parser.add_argument('--max-analysis-frames', type=int, default=120)
    parser.add_argument('--roi', help='x,y,width,height in source pixels')
    parser.add_argument('--annotations', help='Optional existing complete temporal interval ground truth JSON')
    parser.add_argument('--output', help='Output directory, unique by default')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from visionguard.inference.video_inference import VideoInspectionPipeline
    source = Path(args.video).parent / 'manifest.json'
    provenance = json.loads(source.read_text(encoding='utf-8')) if source.exists() else None
    annotations = json.loads(Path(args.annotations).read_text(encoding='utf-8')) if args.annotations else None
    folder = Path(args.output or f'artifacts/VID_{uuid.uuid4().hex[:12]}')
    pipeline = VideoInspectionPipeline(model_name=args.model, category=args.category,
        sampling_interval=args.sampling_interval, max_analysis_frames=args.max_analysis_frames,
        roi=tuple(map(int, args.roi.split(','))) if args.roi else None)
    result = pipeline.process_video(args.video, folder / 'annotated.mp4', args.max_duration, provenance, annotations)
    print(json.dumps(dict(metadata=result['metadata'], metrics=result['metrics']), indent=2))
    print('Artifacts:', folder)
