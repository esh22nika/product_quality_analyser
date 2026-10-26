"""Explicitly synthetic inspection sequence from existing public benchmark test images."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def make_demo(root, category, output_root="datasets/public_videos", dataset="mvtec_ad"):
    folder = Path(output_root) / f"synthetic_{dataset}_{category}"
    folder.mkdir(parents=True, exist_ok=True)
    base = Path(root) / category / "test"
    good = sorted((base / "good").glob("*.*"))[:2]
    bad = sorted(p for p in base.glob("*/*.*") if p.parent.name != "good")[:2]
    if not good or not bad:
        raise FileNotFoundError(f"Need existing normal and anomalous benchmark test images under {base}")
    sequence = good + bad + good[:1]
    output = folder / "synthetic_demo.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 548))
    if not writer.isOpened():
        raise RuntimeError("MP4 codec unavailable")
    try:
        for path in sequence:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable image: {path}")
            image = cv2.resize(image, (512, 512))
            for frame in range(20):
                # A mild zoom creates a simulated camera motion, not a real temporal benchmark.
                offset = frame // 5
                product = cv2.resize(image[offset:512-offset or None, offset:512-offset or None], (512, 512))
                canvas = np.zeros((548, 512, 3), dtype=np.uint8)
                canvas[36:] = product
                cv2.putText(canvas, "SYNTHETIC IMAGE DEMO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
                writer.write(canvas)
    finally:
        writer.release()
    manifest = dict(title=f"Synthetic {category} inspection", video=output.name, dataset=dataset,
                    category=category, sequence_type="synthetic_image_sequence", annotation_type="none",
                    quantitative_evaluation=False, recommended_roi=[0, 36, 512, 512], source_images=[str(p.resolve()) for p in sequence],
                    changes="Repeated test images with mild zoom; not real video; not a video accuracy benchmark.",
                    license="Original benchmark licence applies", source_page="https://www.mvtec.com/research-teaching/datasets/mvtec-ad" if dataset == "mvtec_ad" else "https://github.com/amazon-science/spot-diff")
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='mvtec_anomaly_detection')
    parser.add_argument('--category', default='bottle')
    parser.add_argument('--dataset', choices=['mvtec_ad', 'visa'], default='mvtec_ad')
    parser.add_argument('--output', default='datasets/public_videos')
    args = parser.parse_args()
    print(make_demo(args.root, args.category, args.output, args.dataset))
