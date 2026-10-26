"""Local inspection workspace with serialized background inference and playable media."""
import copy
import json
import logging
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import pandas as pd
import torch
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from visionguard.data.categories import INSPECTION_CATEGORIES, MVTEC_CATEGORIES
from visionguard.data.public_videos import available_videos
from visionguard.data.validation import DatasetValidator
from visionguard.inference.image_inference import ImageInspectionPipeline
from visionguard.inference.video_inference import VideoInspectionPipeline
from visionguard.inference.video_encoding import browser_encode
from visionguard.storage.export import export_image_result
from visionguard.storage.repository import InspectionRepository

ROOT = Path(__file__).resolve().parents[3]

def create_flask_app(test_config=None):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.update(MAX_CONTENT_LENGTH=2*1024**3, ARTIFACT_ROOT=ROOT/'artifacts')
    if test_config:
        app.config.update(test_config)
    artifacts = Path(app.config['ARTIFACT_ROOT']).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    repo = InspectionRepository(artifacts/'visionguard.db')
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inspection')
    jobs, jobs_lock, preview_lock = {}, threading.Lock(), threading.Lock()
    app.extensions.update(inspection_jobs=jobs, inspection_executor=executor)
    sources = {Path(p).parent.name: (Path(p).resolve(), info) for p, info in available_videos(ROOT/'datasets/public_videos').values()
               if info.get('category') in MVTEC_CATEGORIES}
    example_root = ROOT/'mvtec_anomaly_detection/bottle/test'
    examples = {name: next((example_root/sub).glob('*.png'), None) for name, sub in [('normal','good'), ('defective','broken_large')]}

    def media(path):
        return '/media/artifacts/' + Path(path).resolve().relative_to(artifacts).as_posix()

    def records(df):
        return json.loads(df.to_json(orient='records'))

    def update(job_id, **values):
        with jobs_lock:
            jobs[job_id].update(values)

    def image_result(path, cat, model, threshold):
        result = ImageInspectionPipeline(model_name=model, category=cat).inspect(path, decision_threshold=threshold, heatmap_scale='relative')
        if not result['success']:
            raise ValueError(result.get('error', 'Image inspection failed'))
        folder = artifacts/f'INSP_{uuid.uuid4().hex[:12]}'
        archive = export_image_result(result, folder)
        repo.insert_inspection(folder.name, path.name, 'image', cat, model, result['image_score'], result['predicted_label'], artifacts_path=str(folder))
        return dict(id=folder.name, score=result['image_score'], decision=result['predicted_label'], inference_ms=result['inference_ms'],
            region_analysis=result['region_analysis'], quality=result['quality'], model=model, category=cat,
            images={key: media(folder/f'{name}.png') for key,name in [('original','original_image'),('heatmap','heatmap'),('overlay','annotated_overlay'),('mask','thresholded_mask')]},
            downloads={'Inspection package': media(archive), 'JSON report': media(folder/'report.json')})

    def saved_video(folder, gallery=None):
        report = json.loads((folder/'inspection_report.json').read_text(encoding='utf-8'))
        frames = pd.read_csv(folder/'frame_scores.csv')
        if gallery is not None:
            for i, item in enumerate(gallery):
                cv2.imwrite(str(folder/f'preview_{i}.jpg'), cv2.cvtColor(item['overlay'], cv2.COLOR_RGB2BGR))
        elif not (folder/'preview_0.jpg').exists():
            cap = cv2.VideoCapture(str(folder/'annotated.mp4'))
            try:
                for i, row in enumerate(frames.head(12).itertuples()):
                    cap.set(cv2.CAP_PROP_POS_MSEC, row.timestamp_s*1000)
                    ok, frame = cap.read()
                    if ok:
                        cv2.imwrite(str(folder/f'preview_{i}.jpg'), frame)
            finally:
                cap.release()
        return dict(id=folder.name, video_url=media(folder/'annotated.mp4'), metadata=report['metadata'], metrics=report['metrics'],
            events=report.get('events',[]), frames=records(frames),
            gallery=[dict(url=media(folder/f'preview_{i}.jpg'), timestamp_s=row.timestamp_s, score=row.anomaly_score)
                     for i,row in enumerate(frames.head(12).itertuples()) if (folder/f'preview_{i}.jpg').exists()],
            downloads={label:media(folder/file) for label,file in [('Annotated MP4','annotated.mp4'),('Frame scores CSV','frame_scores.csv'),('Events CSV','inspection_events.csv'),('JSON report','inspection_report.json')]})

    def work(job_id, kind, paths, options):
        update(job_id, state='running', message='Loading trained checkpoint…', progress=.02)
        try:
            cat, model, threshold = options['category'], options['model'], options['threshold']
            if kind == 'video':
                folder = artifacts/f'VID_{uuid.uuid4().hex[:12]}'
                pipeline = VideoInspectionPipeline(model_name=model, category=cat, decision_threshold=threshold,
                    sampling_interval=options['sampling_interval'], consecutive_frames=options['consecutive_frames'], roi=options['roi'])
                result = pipeline.process_video(paths[0], folder/'annotated.mp4', max_duration_s=options['seconds'],
                    provenance=options['provenance'], annotations=options['annotations'],
                    on_progress=lambda fraction,message: update(job_id, progress=min(fraction,.99), message=message))
                repo.insert_inspection(folder.name, paths[0].name, 'video', cat, model, float(result['frame_dataframe'].anomaly_score.max()), 'REVIEW_REQUIRED', artifacts_path=str(folder))
                payload = saved_video(folder, result['suspicious_gallery'])
            elif kind == 'batch':
                rows = []
                for i,path in enumerate(paths):
                    update(job_id, progress=i/len(paths), message=f'Inspecting image {i+1} of {len(paths)}…')
                    try:
                        result = image_result(path, cat, model, threshold)
                        rows.append(dict(filename=path.name,status='Completed',score=result['score'],decision=result['decision'],report=result['downloads']['JSON report']))
                    except Exception as exc:
                        rows.append(dict(filename=path.name,status='Failed',error=str(exc)))
                folder = artifacts/f'BATCH_{job_id}'
                folder.mkdir()
                pd.DataFrame(rows).to_csv(folder/'results.csv',index=False)
                payload = dict(rows=rows, downloads={'Batch CSV':media(folder/'results.csv')})
            else:
                payload = image_result(paths[0],cat,model,threshold)
            update(job_id,state='completed',progress=1,message='Inspection complete',result=payload)
        except Exception as exc:
            logging.getLogger(__name__).exception('Inspection job %s failed',job_id)
            update(job_id,state='failed',message=str(exc),error=str(exc))

    def number(name, default, low, high, integer=False):
        value = float(request.form.get(name,default))
        if not math.isfinite(value) or not low <= value <= high or (integer and not value.is_integer()):
            raise ValueError(f'{name} must be between {low} and {high}' + (' (whole number)' if integer else ''))
        return int(value) if integer else value

    def upload(file, kind):
        suffix = Path(file.filename or '').suffix.lower()
        allowed = {'.mp4','.avi','.mov','.webm'} if kind == 'video' else {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
        if suffix not in allowed:
            raise ValueError(f'Unsupported {kind} file: {suffix}')
        folder = artifacts/'uploads'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        path = folder/(secure_filename(file.filename) or ('input'+suffix))
        file.save(path)
        if kind != 'video':
            if path.stat().st_size > 40*1024**2:
                raise ValueError('Images must be under 40 MB')
            with Image.open(path) as image:
                if image.width*image.height > 40_000_000:
                    raise ValueError('Images must be under 40 megapixels')
                image.verify()
        return path

    @app.get('/')
    def index():
        return render_template('index.html',device='GPU' if torch.cuda.is_available() else 'CPU')

    @app.get('/api/config')
    def config():
        availability = {model:[cat for cat in INSPECTION_CATEGORIES if (ROOT/'checkpoints'/f'{model}_{cat}.pt').is_file() or
            (model == 'efficientad' and (ROOT/'checkpoints/public_efficientad'/cat/'model.ckpt').is_file())] for model in ['efficientad','patchcore']}
        return jsonify(models=availability,device='GPU' if torch.cuda.is_available() else 'CPU',
            sources=[dict(id=key,title=info['title'],sequence_type=info['sequence_type'],preview_url=url_for('source_video',source_id=key),provenance=info) for key,(_,info) in sources.items()],
            examples={key:url_for('example',name=key) for key,value in examples.items() if value})

    @app.get('/media/examples/<name>')
    def example(name):
        if examples.get(name) is None:
            abort(404)
        return send_file(examples[name],conditional=True)

    @app.get('/media/sources/<source_id>')
    def source_video(source_id):
        if source_id not in sources:
            abort(404)
        import shutil
        original,_ = sources[source_id]
        target = artifacts/'web_previews'/f'{source_id}.mp4'
        with preview_lock:
            if not target.exists() or target.stat().st_mtime < original.stat().st_mtime:
                target.parent.mkdir(exist_ok=True)
                temporary = target.with_suffix('.working.mp4')
                shutil.copyfile(original,temporary)
                browser_encode(temporary)
                temporary.replace(target)
        return send_file(target,mimetype='video/mp4',conditional=True)

    @app.get('/media/artifacts/<path:filename>')
    def artifact(filename):
        path = (artifacts/filename).resolve()
        if not path.is_relative_to(artifacts) or not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.mp4','.csv','.json','.zip'}:
            abort(404)
        if not path.parts[len(artifacts.parts)].startswith(('INSP_','VID_','BATCH_')):
            abort(404)
        return send_file(path,conditional=True,as_attachment=path.suffix in {'.json','.csv','.zip'})

    @app.post('/api/jobs')
    def start_job():
        with jobs_lock:
            if sum(j['state'] in {'queued','running'} for j in jobs.values()) >= 2:
                return jsonify(error='An inspection is already queued. Wait for it to finish.'),409
        kind = request.form.get('kind','image')
        if kind not in {'image','video','batch'}:
            raise ValueError('Unknown inspection type')
        model,cat = request.form.get('model','efficientad'),request.form.get('category','bottle')
        if model not in {'efficientad','patchcore'} or cat not in INSPECTION_CATEGORIES:
            raise ValueError('Choose a supported model and product category')
        options = dict(model=model,category=cat,threshold=number('threshold',.5,0,1))
        if kind == 'video':
            source_id = request.form.get('source_id')
            if source_id and source_id not in sources:
                raise ValueError('Unknown public video source')
            source_category = sources[source_id][1].get('category') if source_id else None
            if source_category and cat != source_category:
                raise ValueError(f"This footage contains {source_category}. Select the matching product category.")
            if source_id and sources[source_id][1].get('preview_only'):
                raise ValueError('This source needs an AeBAD blade-trained detector. Bottle and PCB checkpoints are unsuitable. Preview is available; no training has been started.')
            options.update(seconds=number('seconds',10,1,60),sampling_interval=number('sampling_interval',15,1,120,True),
                consecutive_frames=number('consecutive_frames',3,1,10,True),roi=None,annotations=None,
                provenance=copy.deepcopy(sources[source_id][1]) if source_id else dict(sequence_type='uploaded_video',annotation_type='none'))
            roi = request.form.get('roi','').strip()
            if roi:
                options['roi'] = tuple(map(int,roi.split(',')))
                if len(options['roi']) != 4:
                    raise ValueError('ROI needs four numbers: x,y,width,height')
            annotation = request.files.get('annotations')
            if annotation and annotation.filename:
                options['annotations'] = json.loads(annotation.stream.read(2*1024**2))
            paths = [sources[source_id][0]] if source_id else []
        else:
            example_id = request.form.get('example')
            paths = [examples[example_id]] if example_id in examples and examples[example_id] else []
        if not paths:
            files = [file for file in request.files.getlist('files') if file.filename]
            if not files or (kind != 'batch' and len(files) != 1) or len(files)>50:
                raise ValueError('Choose one file, or up to 50 images for a batch')
            paths = [upload(file,kind) for file in files]
        job_id = uuid.uuid4().hex
        with jobs_lock:
            for old in list(jobs):
                if len(jobs)<100:
                    break
                if jobs[old]['state'] in {'completed','failed'}:
                    jobs.pop(old)
            jobs[job_id] = dict(id=job_id,kind=kind,state='queued',progress=0,message='Queued for inspection…')
        executor.submit(work,job_id,kind,paths,options)
        return jsonify(id=job_id),202

    @app.get('/api/jobs/<job_id>')
    def job_status(job_id):
        with jobs_lock:
            if job_id not in jobs:
                return jsonify(error='Job is no longer in memory. Load the last completed video.'),404
            return jsonify(copy.deepcopy(jobs[job_id]))

    @app.get('/api/videos/latest')
    def latest_video():
        for report in sorted(artifacts.glob('VID_*/inspection_report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            if all((report.parent/file).exists() for file in ['annotated.mp4','frame_scores.csv']):
                return jsonify(saved_video(report.parent))
        return jsonify(error='No completed video yet. Run the demo first.'),404

    @app.get('/api/history')
    def history():
        return jsonify(records(repo.list_inspections(limit=100)))

    @app.get('/api/datasets')
    def datasets():
        mvtec = ROOT/'datasets/mvtec_ad'
        return jsonify(mvtec=DatasetValidator.validate_mvtec_ad(mvtec if mvtec.exists() else ROOT/'mvtec_anomaly_detection'),visa=DatasetValidator.validate_visa(ROOT/'datasets/visa'))

    @app.get('/api/evaluation')
    def evaluation():
        rows=[]
        for path in (ROOT/'reports').glob('*.json'):
            try:
                data=json.loads(path.read_text(encoding='utf-8'))
                if data.get('schema_version') != 2:
                    continue
                info,metrics=data.get('dataset_info',{}),data.get('metrics',{})
                rows.append(dict(model=data.get('model_info',{}).get('name'),category=info.get('category'),samples=metrics.get('sample_count'),
                    protocol=info.get('split'),corruption=info.get('corruption','none'),image_auroc=metrics.get('auroc'),pixel_auroc=metrics.get('pixel_auroc'),iou=metrics.get('iou')))
            except (ValueError,OSError):
                continue
        return jsonify(rows)

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc,HTTPException):
            return jsonify(error=exc.description),exc.code
        if isinstance(exc,(ValueError,KeyError,TypeError)):
            return jsonify(error=str(exc)),400
        logging.getLogger(__name__).exception('Flask request failed')
        return jsonify(error=str(exc)),500

    return app
