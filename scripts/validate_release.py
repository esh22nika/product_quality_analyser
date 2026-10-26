"""Small real-checkpoint release checks; no training and no invented benchmark claims."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    import torch
    torch.set_num_threads(4)
    from visionguard.evaluation.benchmark import evaluate
    from visionguard.models.loader import ModelManager
    from visionguard.inference.image_inference import ImageInspectionPipeline
    from visionguard.storage.export import export_image_result
    from visionguard.inference.video_inference import VideoInspectionPipeline
    from visionguard.evaluation.reports import BenchmarkReporter
    import cv2
    started = time.perf_counter()
    rows = []
    for dataset, category, root, limit in [('mvtec_ad', 'bottle', 'mvtec_anomaly_detection', 5),
                                            ('visa', 'pcb1', 'datasets/visa', 2)]:
        for model in ['patchcore', 'efficientad']:
            path, metrics = evaluate(model,dataset,category,root,limit_per_class=limit)
            rows.append(dict(model=model,category=category,dataset=dataset,report=str(path),**metrics))
            print('Verified image evaluation:',model,dataset,category,flush=True)
    BenchmarkReporter().save_comparison_table(rows,'release_image_comparison.csv')
    for model in ['patchcore','efficientad']:
        clean = None
        robustness = []
        for corruption in ['none','brightness','blur']:
            path, metrics = evaluate(model,'mvtec_ad','bottle','mvtec_anomaly_detection',limit_per_class=2,corruption=corruption)
            if corruption == 'none':
                clean = metrics['auroc']
            robustness.append(dict(model=model,corruption=corruption,auroc=metrics['auroc'],
                                   auroc_change_from_clean=metrics['auroc']-clean,report=str(path)))
        BenchmarkReporter().save_comparison_table(robustness,f'release_robustness_{model}.csv')
        print('Verified corruption smoke sweep:',model,flush=True)
    result = ImageInspectionPipeline('efficientad','bottle').inspect('mvtec_anomaly_detection/bottle/test/good/000.png')
    assert result['success'], result.get('error')
    archive = export_image_result(result, ROOT/'artifacts/INSP_release_sample')
    videos = []
    for source, duration, interval in [('commons_gigaset_conveyor',2,15),('synthetic_mvtec_ad_bottle',10,20)]:
        folder = ROOT/'datasets/public_videos'/source
        provenance=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        output=ROOT/'artifacts'/f'VID_release_{source}'/'annotated.mp4'
        pipeline=VideoInspectionPipeline('efficientad','bottle',sampling_interval=interval,consecutive_frames=1)
        inspected=pipeline.process_video(folder/provenance['video'],output,duration,provenance)
        assert inspected['metrics']['event_precision'] is None
        cap=cv2.VideoCapture(str(output))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        cap.release()
        videos.append(dict(source=source,report=str(output.parent/'inspection_report.json'),
                           processed_frames=inspected['metadata']['processed_frames'],
                           evaluation_mode=inspected['metrics']['evaluation_mode']))
        print('Verified real-model H.264 video/export:',source,flush=True)
    ModelManager.clear_cache()
    record=dict(training_performed=False,elapsed_s=time.perf_counter()-started,
                dependencies={name:metadata.version(name) for name in ['torch','torchvision','anomalib','gradio','lightning','imageio-ffmpeg']},
                cuda_available=torch.cuda.is_available(),image_evaluations=rows,videos=videos,image_archive=archive,
                limitations=['Image results are small deterministic test subsets, not full benchmarks.',
                             'Real conveyor footage is unlabeled and out of domain; no video accuracy claim.',
                             'Synthetic video is a benchmark-image demonstration, not real motion performance.'])
    Path('reports/release_verification.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    print('Release verification saved to reports/release_verification.json',flush=True)

if __name__ == '__main__':
    main()
