"""Verified public demonstration footage; never invent defect annotations."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import cv2

CONVEYOR = {
    "id": "commons_gigaset_conveyor", "title": "Gigaset pneumatic conveyor belt",
    "source_page": "https://commons.wikimedia.org/wiki/File:Gigaset_Cordless_Telephone_Production_VII_-_Pneumatic_Conveyor_Belt.webm",
    "download_url": "https://upload.wikimedia.org/wikipedia/commons/3/36/Gigaset_Cordless_Telephone_Production_VII_-_Pneumatic_Conveyor_Belt.webm",
    "license": "CC BY 3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/",
    "attribution": "Sounds of Changes; video/photography Inke Pickhardt; sound Kathinka Engels",
    "original_sha1": "ff3737d29a4ca66e51368bbe652eb3747dd117ae",
    "sequence_type": "real_video", "annotation_type": "none", "quantitative_evaluation": False,
    "suitability": "Real moving packages on a conveyor; qualitative pipeline demo only. Existing MVTec/VisA models are out of domain.",
}


def prepare_conveyor(root="datasets/public_videos"):
    folder = Path(root) / CONVEYOR["id"]
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / "original.webm"
    if not original.exists():
        request = urllib.request.Request(CONVEYOR["download_url"], headers={"User-Agent": "VisionGuard/0.1 research dataset preparation"})
        temporary = original.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if hashlib.sha1(temporary.read_bytes()).hexdigest() != CONVEYOR["original_sha1"]:
            raise ValueError("Public source checksum changed. Reverify source before using this download.")
        os.replace(temporary, original)
    if hashlib.sha1(original.read_bytes()).hexdigest() != CONVEYOR["original_sha1"]:
        raise ValueError("The cached original does not match the verified source.")
    output = folder / "conveyor.mp4"
    capture = cv2.VideoCapture(str(original))
    writer = None
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV cannot decode this public WebM; install an FFmpeg-enabled OpenCV build.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            raise ValueError("Invalid source frame rate")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            width = min(640, w)
            height = max(2, int(h * width / w) // 2 * 2)
            frame = cv2.resize(frame, (width, height))
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("MPEG-4 video writer unavailable")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if count == 0:
        raise ValueError("Source contains no decodable video frames")
    manifest = dict(CONVEYOR, video="conveyor.mp4", frames=count, fps=fps, duration_s=count / fps,
                    changes="Resized to <=640px width, transcoded to MP4, audio omitted; no labels added.")
    with output.open("rb") as stream:
        manifest["prepared_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (folder / "ATTRIBUTION.txt").write_text(
        f"{CONVEYOR['title']}\n{CONVEYOR['attribution']}\n{CONVEYOR['license_url']}\n{CONVEYOR['source_page']}\n{manifest['changes']}\n", encoding="utf-8")
    return output


def available_videos(root="datasets/public_videos"):
    result = {}
    for path in Path(root).glob("*/manifest.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        video = path.parent / item["video"]
        if video.is_file():
            result[f"{item['title']} [{item['sequence_type']}; qualitative]"] = (str(video), item)
    return result
