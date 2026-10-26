"""Download pinned, publicly shared trained EfficientAD category checkpoints."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
MVTEC_REPO = 'micguida1/mvtec-anomaly-checkpoints'
MVTEC_REVISION = 'a0d5049619c09cdd06e0fb22f82c906b53355d2f'
PCB_REPO = 'thangkt/visionqc-efficientad-medium-pcb1'
PCB_REVISION = '2e7b979c09ca77188acea2cd550f3d31493dd7df'

def download(job):
    category, repo, revision, filename = job
    output = ROOT / 'checkpoints' / 'public_efficientad' / category / 'model.ckpt'
    output.parent.mkdir(parents=True, exist_ok=True)
    record_path = output.parent / 'source.json'
    if output.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get('revision') == revision and record.get('bytes') == output.stat().st_size:
            print(f'Already downloaded: {category}', flush=True)
            return record
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}'
    temporary = output.with_suffix('.ckpt.part')
    print(f'Downloading trained EfficientAD: {category}', flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as handle:
        expected = response.headers.get('Content-Length')
        count = 0
        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
            handle.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    if expected is not None and count != int(expected):
        raise RuntimeError(f'Incomplete download: {category}: {count}/{expected}')
    os.replace(temporary, output)
    record = dict(category=category, repo=repo, revision=revision, filename=filename,
                  bytes=count, sha256=digest.hexdigest(), url=url)
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Saved {category}: {count / 1e6:.1f} MB', flush=True)
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', default=CATEGORIES + ['pcb1'])
    args = parser.parse_args()
    jobs = []
    for category in args.categories:
        if category in CATEGORIES:
            jobs.append((category, MVTEC_REPO, MVTEC_REVISION,
                         f'EfficientAd/MVTecAD/{category}/latest/weights/lightning/model.ckpt'))
        elif category == 'pcb1':
            jobs.append((category, PCB_REPO, PCB_REVISION, 'model-best.ckpt'))
        else:
            parser.error(f'Unsupported category: {category}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, jobs))

if __name__ == '__main__':
    main()
