"""Prepare a real, ordered AeBAD-V frame sequence from the official public archive.

Downloads only the ZIP byte ranges containing the chosen video. Original frame
labels are retained; playback FPS is chosen because source timestamps are absent.
No model training or quantitative temporal evaluation is performed.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
FILE_ID = '14wkZAFFeudlg0NMFLsiGwS0E593b-lNo'
SOURCE = 'https://github.com/zhangzilongc/MMR'
DOWNLOAD = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download'

def download_location():
    with urllib.request.urlopen(DOWNLOAD, timeout=60) as response:
        if 'text/html' not in response.headers.get('Content-Type',''):
            raise RuntimeError('Unexpected Drive response; use the official archive link')
        text = response.read(100000).decode()
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', text))
    form = re.search(r'<form[^>]*action="([^"]+)"', text)
    if not form or not fields:
        raise RuntimeError('Drive download unavailable; see the official source link')
    location = html.unescape(form.group(1))
    if urllib.parse.urlparse(location).hostname != 'drive.usercontent.google.com':
        raise RuntimeError('Unexpected download host')
    url = location + '?' + urllib.parse.urlencode(fields)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Range':'bytes=0-0'}), timeout=60) as response:
        if response.status != 206:
            raise RuntimeError('Server must support partial downloads')
        size = int(response.headers['Content-Range'].split('/')[-1])
        response.read(1)
    return url,size

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.pos=url,size,0
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def tell(self):
        return self.pos
    def read(self,size=-1):
        end=self.size-1 if size<0 else min(self.size-1,self.pos+size-1)
        if end<self.pos:
            return b''
        if end-self.pos>200*1024**2:
            raise ValueError('Refusing an unexpectedly large range')
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req,timeout=120) as response:
            if response.status!=206:
                raise RuntimeError('Server ignored bounded download request')
            data=response.read(end-self.pos+1)
        if len(data)!=end-self.pos+1:
            raise RuntimeError('Incomplete archive byte range')
        self.pos+=len(data)
        return data

def prepare(video='video1',fps=15,output_root=ROOT/'datasets/public_videos'):
    if video not in {'video1','video2','video3'} or not 1<=fps<=60:
        raise ValueError('Use video1/video2/video3 and playback FPS 1–60')
    folder=Path(output_root)/f'aebad_v_{video}'
    folder.mkdir(parents=True,exist_ok=True)
    raw=folder/'original_frames'
    raw.mkdir(exist_ok=True)
    url,size=download_location()
    remote=RemoteZip(url,size)
    with zipfile.ZipFile(remote) as archive:
        entries=[i for i in archive.infolist() if i.filename.startswith(f'AeBAD/AeBAD_V/test/{video}/') and Path(i.filename).suffix.lower()=='.jpg' and Path(i.filename).stem.isdigit()]
    if not entries:
        raise ValueError('Expected official ordered video frames not found')
    labels=[]
    for kind in ['good','anomaly']:
        group=[i for i in entries if Path(i.filename).parent.name==kind]
        missing=[i for i in group if not (raw/Path(i.filename).name).exists()]
        if missing:
            start=min(i.header_offset for i in missing)
            end=max(i.header_offset+30+len(i.filename.encode())+65535+i.compress_size for i in missing)
            remote.seek(start)
            print(f'Downloading {video}/{kind}: {(end-start)/1e6:.1f} MB',flush=True)
            block=remote.read(end-start)
            for item in missing:
                offset=item.header_offset-start
                header=struct.unpack('<4s5H3I2H',block[offset:offset+30])
                if header[0]!=b'PK\x03\x04':
                    raise ValueError('Invalid ZIP local header')
                offset+=30+header[-2]+header[-1]
                compressed=block[offset:offset+item.compress_size]
                data=zlib.decompress(compressed,-15) if item.compress_type==8 else compressed if item.compress_type==0 else None
                if data is None or len(data)!=item.file_size or zlib.crc32(data)!=item.CRC:
                    raise ValueError('Original image checksum mismatch')
                (raw/Path(item.filename).name).write_bytes(data)
        for item in group:
            labels.append(dict(source_frame=int(Path(item.filename).stem),filename=Path(item.filename).name,label=int(kind=='anomaly'),source_path=item.filename))
    labels.sort(key=lambda row:row['source_frame'])
    if len({row['source_frame'] for row in labels})!=len(labels):
        raise ValueError('Frame IDs are not unique')
    gaps=[b['source_frame']-a['source_frame'] for a,b in zip(labels,labels[1:]) if b['source_frame']-a['source_frame']!=1]
    if gaps:
        raise ValueError('Sequence has missing frames; refusing to imply continuous footage')
    import cv2
    from visionguard.inference.video_encoding import browser_encode
    output=folder/'inspection.mp4'
    first=cv2.imread(str(raw/labels[0]['filename']))
    height,width=first.shape[:2]
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    try:
        for position,row in enumerate(labels):
            frame=cv2.imread(str(raw/row['filename']))
            if frame is None or frame.shape[:2]!=(height,width):
                raise ValueError('Invalid source image')
            writer.write(frame)
            row['playback_frame']=position
            row['playback_timestamp_s']=position/fps
            row['sha256']=hashlib.sha256((raw/row['filename']).read_bytes()).hexdigest()
    finally:
        writer.release()
    browser_encode(output)
    with (folder/'frame_labels.csv').open('w',newline='',encoding='utf-8') as stream:
        writer_csv=csv.DictWriter(stream,fieldnames=list(labels[0]))
        writer_csv.writeheader();writer_csv.writerows(labels)
    manifest=dict(title=f'AeBAD-V blade surface inspection — {video}',video=output.name,dataset='aebad_v',
        sequence_type='real_frame_sequence',annotation_type='frame_labels',quantitative_evaluation=False,
        category='aebad_blade',required_category='aebad_blade',preview_only=True,
        suitability='Real close-up blade surface sequence. Requires a blade-trained detector; bottle/PCB models are unsuitable.',
        changes='Original ordered 800px JPEG frames transcoded to H.264. Playback FPS is chosen, not original timing. Source frame labels preserved separately.',
        license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',source_page=SOURCE,
        attribution='Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, Xuefeng Chen — AeBAD dataset (2023).',
        official_archive=f'https://drive.google.com/file/d/{FILE_ID}/view',official_archive_size=size,
        frames=len(labels),fps=fps,duration_s=len(labels)/fps,resolution=[width,height],
        timing_status='Chosen playback FPS; no original capture timestamps verified',labels_file='frame_labels.csv',
        prepared_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (folder/'ATTRIBUTION.txt').write_text(manifest['attribution']+'\n'+SOURCE+'\n'+manifest['license_url']+'\n'+manifest['changes'],encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',choices=['video1','video2','video3'],default='video1')
    parser.add_argument('--fps',type=int,default=15)
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/public_videos')
    args=parser.parse_args()
    prepare(args.video,args.fps,args.output)
