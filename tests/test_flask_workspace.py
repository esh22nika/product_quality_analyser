import io
import json
import time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from visionguard.web.app_flask import create_flask_app

@pytest.fixture
def web(tmp_path, monkeypatch):
    # Exercise the API without requiring downloaded benchmark media or weights.
    monkeypatch.setattr('visionguard.web.app_flask.ROOT', tmp_path)
    checkpoint = tmp_path/'checkpoints/public_efficientad/bottle/model.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture; inference is mocked')
    for name, category, sequence in [('bottle_real','bottle','real_video'),
                                     ('bottle_synthetic','bottle','synthetic_image_sequence'),
                                     ('blade','aebad_blade','real_frame_sequence')]:
        folder=tmp_path/'datasets/public_videos'/name
        folder.mkdir(parents=True)
        (folder/'video.mp4').write_bytes(b'fixture; only catalogue metadata is read')
        (folder/'manifest.json').write_text(json.dumps(dict(title=name, category=category,
            sequence_type=sequence, video='video.mp4')), encoding='utf-8')
    app=create_flask_app({'TESTING':True,'ARTIFACT_ROOT':tmp_path/'artifacts'})
    yield app,app.test_client()
    app.extensions['inspection_executor'].shutdown(wait=True)

def wait(client,job_id):
    for _ in range(100):
        result=client.get('/api/jobs/'+job_id).get_json()
        if result['state'] in {'failed','completed'}:
            return result
        time.sleep(.02)
    raise AssertionError('Job did not finish')

def png():
    stream=io.BytesIO()
    Image.new('RGB',(32,32),'white').save(stream,format='PNG')
    stream.seek(0)
    return stream

def test_workspace_and_available_models(web):
    _,client=web
    page=client.get('/')
    assert page.status_code==200
    assert b'color-scheme' in page.data and b'video-player' in page.data
    assert b'cdn.jsdelivr' not in page.data
    config=client.get('/api/config').get_json()
    assert 'bottle' in config['models']['efficientad']
    assert len(config['sources'])>=2
    assert client.get('/static/workspace.css').status_code==200

def test_upload_and_parameters_rejected_before_job(web):
    _,client=web
    assert client.post('/api/jobs',data={'kind':'image'}).status_code==400
    assert client.post('/api/jobs',data={'kind':'video','seconds':'nan'}).status_code==400
    assert client.post('/api/jobs',data={'kind':'video','sampling_interval':'1.5'}).status_code==400
    assert client.post('/api/jobs',data={'kind':'image','files':(io.BytesIO(b'x'),'bad.exe')}).status_code==400
    assert client.get('/media/artifacts/visionguard.db').status_code==404
    assert client.get('/media/artifacts/../requirements.txt').status_code==404

def test_async_image_outputs_and_failure(web,monkeypatch):
    app,client=web
    def inspect(self,path,**kwargs):
        image=np.full((32,32,3),127,np.uint8)
        amap=np.zeros((32,32),np.float32)
        return dict(success=True,original_image=image,heatmap=image,annotated_overlay=image,thresholded_mask=amap.astype('uint8'),
            anomaly_map=amap,raw_anomaly_map=amap,image_score=.4,predicted_label='LIKELY_NORMAL',inference_ms=1,
            quality={'warnings':[]},region_analysis={'estimated_anomalous_area_percent':0})
    monkeypatch.setattr('visionguard.web.app_flask.ImageInspectionPipeline.inspect',inspect)
    response=client.post('/api/jobs',data={'kind':'image','files':(png(),'product.png')})
    assert response.status_code==202
    result=wait(client,response.get_json()['id'])
    assert result['state']=='completed'
    assert client.get(result['result']['images']['heatmap']).status_code==200
    assert client.get(result['result']['downloads']['Inspection package']).status_code==200
    assert client.get('/api/history').get_json()[0]['input_filename']=='product.png'
    monkeypatch.setattr('visionguard.web.app_flask.ImageInspectionPipeline.inspect',lambda *a,**kw:dict(success=False,error='checkpoint unavailable'))
    response=client.post('/api/jobs',data={'kind':'image','files':(png(),'product.png')})
    failed=wait(client,response.get_json()['id'])
    assert failed['state']=='failed' and 'checkpoint unavailable' in failed['error']
    assert 'result' not in failed

def test_media_supports_browser_range_requests(web):
    app,client=web
    folder=Path(app.config['ARTIFACT_ROOT'])/'VID_fixture'
    folder.mkdir()
    (folder/'annotated.mp4').write_bytes(bytes(range(100)))
    response=client.get('/media/artifacts/VID_fixture/annotated.mp4',headers={'Range':'bytes=10-19'})
    assert response.status_code==206
    assert response.data==bytes(range(10,20))
    assert response.headers['Content-Type']=='video/mp4'

def test_public_sources_match_mvtec_and_reject_wrong_category(web):
    from visionguard.data.categories import MVTEC_CATEGORIES
    _,client=web
    config=client.get('/api/config').get_json()
    assert config['sources']
    assert all(item['provenance'].get('category') in MVTEC_CATEGORIES for item in config['sources'])
    assert all('AeBAD' not in item['title'] and 'Gigaset' not in item['title'] for item in config['sources'])
    source=next(item for item in config['sources'] if item['sequence_type']=='real_video')
    response=client.post('/api/jobs',data={'kind':'video','source_id':source['id'],'category':'hazelnut','model':'efficientad'})
    assert response.status_code==400 and 'matching product category' in response.get_json()['error']
