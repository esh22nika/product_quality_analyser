"""Download licensed real bottle-category demonstration footage; never invent labels."""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visionguard.inference.video_encoding import browser_encode

SOURCE=dict(id='pexels_bottle_line_10416677',title='Real glass bottles on a production line',category='bottle',
    source_page='https://www.pexels.com/video/production-line-in-close-up-10416677/',
    download_url='https://videos.pexels.com/video-files/10416677/10416677-hd_1280_720_50fps.mp4',
    license='Pexels License',license_url='https://www.pexels.com/license/',attribution='Thanh Văn / Pexels, video 10416677',
    sequence_type='real_video',annotation_type='none',quantitative_evaluation=False,
    domain_warning='Bottle category match only: capped, angled conveyor view differs from MVTec open, top-down bottle images. Existing model accuracy is not validated on this footage.',
    suitability='Visible brown glass bottles moving through a production line. Unlabeled public demonstration footage, not a defect-detection benchmark.',
    recommended_sampling_interval=25)

def prepare(output=ROOT/'datasets/public_videos',local_source=None):
    folder=Path(output)/SOURCE['id'];folder.mkdir(parents=True,exist_ok=True)
    original=folder/'original.mp4'
    if not original.exists():
        if local_source:
            shutil.copyfile(local_source,original)
        else:
            request=urllib.request.Request(SOURCE['download_url'],headers={'User-Agent':'VisionGuard/0.1 research dataset preparation'})
            temporary=folder/'download.part'
            with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as stream:
                if not response.headers.get('Content-Type','').startswith(('video/','application/octet-stream')):
                    raise ValueError('Download did not return a video')
                length=0
                while block:=response.read(1024*1024):
                    length+=len(block)
                    if length>200*1024**2:raise ValueError('Unexpected source size')
                    stream.write(block)
            temporary.replace(original)
    cap=cv2.VideoCapture(str(original))
    fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok,_=cap.read();cap.release()
    if not ok or frames<1 or not 1<=fps<=120:raise ValueError('Source video cannot be decoded')
    target=folder/'bottles.mp4'
    shutil.copyfile(original,target);browser_encode(target)
    manifest=dict(SOURCE,video=target.name,frames=frames,fps=fps,duration_s=frames/fps,resolution=[width,height],
        original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),prepared_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        changes='Re-encoded as H.264/yuv420p for browser playback; audio omitted; original duration, frame order and frame rate retained. No generated footage, defect labels or predictions added.')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (folder/'ATTRIBUTION.txt').write_text('\n'.join([SOURCE['attribution'],SOURCE['source_page'],SOURCE['license_url'],manifest['changes']]),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/public_videos')
    parser.add_argument('--local-source',type=Path,help='Previously downloaded original from the listed source')
    args=parser.parse_args();prepare(args.output,args.local_source)
