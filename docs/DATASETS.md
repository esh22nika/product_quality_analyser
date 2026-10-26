# Dataset preparation

No original recordings or hand-created dataset is required.

## MVTec AD

Official source and terms: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
(CC BY-NC-SA 4.0). Use the existing `mvtec_anomaly_detection/<category>` folders or
place official extracted data under a chosen root. Preserve `train/good`, `test`,
and `ground_truth` and the source licence. The Colab notebook also documents its
pinned public dataset mirror; that mirror is image data, not model weights.

`python scripts/validate_dataset.py --dataset mvtec_ad --root mvtec_anomaly_detection`

## VisA

Official source: https://github.com/amazon-science/spot-diff and
https://registry.opendata.aws/visa/ (CC BY 4.0). The project already has the raw
`datasets/visa` and prepared `datasets/visa/visa_pytorch` layouts. Raw image
evaluation uses the official `split_csv/1cls.csv`; the app's synthetic-video
preparer uses prepared category directories. Anomalib's Visa datamodule can
download the official archive and apply its CSV split if no dataset is installed.
Its root is the parent of `visa_pytorch`, never `visa_pytorch` itself.

`python scripts/validate_dataset.py --dataset visa --root datasets/visa`

## Public real video and optional synthetic demonstration

See PUBLIC_VIDEOS.md for source verification, licence/annotation decisions and
preparation commands. `prepare_commons_conveyor.py` automatically downloads the
licensed real conveyor recording; `make_demo_video.py` optionally creates the
explicitly synthetic benchmark-image sequence. They are separate dataset types.

## Optional MVTec AD 2

https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2 has a different set of
categories and evaluation conditions. It is not a drop-in test set for the current
MVTec AD category checkpoints. It is optional and has not been integrated/evaluated;
the app does not require its restricted/private evaluation subset.
