# Public industrial videos (Dataset D)

The app requires no recordings, photos, or hand-built dataset. Prepared sources appear
in the Video Inspection source selector after restarting the app.

## Verified and integrated: Gigaset conveyor footage

- Source: https://commons.wikimedia.org/wiki/File:Gigaset_Cordless_Telephone_Production_VII_-_Pneumatic_Conveyor_Belt.webm
- Original: 20.116-second real temporal recording, 1280 x 720, moving packaged telephones
  on a pneumatic conveyor. Original size 3,794,865 bytes.
- Licence: CC BY 3.0 per the reviewed file licensing section and structured data.
  The description also mentions CC BY 4.0; the downloader conservatively preserves
  the reviewed CC BY 3.0 attribution and links to the original page.
- Attribution: Sounds of Changes; Inke Pickhardt (video/photography), Kathinka Engels (sound).
- Verification: download SHA-1 checked against the publisher's file checksum, actual
  OpenCV decoding, frame count/FPS recorded in manifest. Audio omitted; resized MP4.
- Annotations: **none**. No claims about which frames have defects; normality is not
  inferred from a filename, absence of labels, or apparent smooth operation.
- Suitability: exercises upload, streaming inference, temporal grouping and exports.
  Existing MVTec/VisA detectors are out of domain for these packages. Heatmaps are
  qualitative visual-unfamiliarity responses, not validated package defects.

```powershell
python scripts/prepare_commons_conveyor.py
python scripts/inspect_video.py datasets/public_videos/commons_gigaset_conveyor/conveyor.mp4 --category bottle --model efficientad --max-duration 5
```

Downloads and conversion preserve source, licence, attribution, original checksum,
and changes in `datasets/public_videos/commons_gigaset_conveyor/manifest.json` and
`ATTRIBUTION.txt`. The original WebM is retained. No media is committed to Git.

## Optional, explicitly synthetic image demonstration

```powershell
python scripts/make_demo_video.py --root mvtec_anomaly_detection --category bottle
# VisA uses the prepared image root, which already contains category/test/good:
python scripts/make_demo_video.py --dataset visa --root datasets/visa/visa_pytorch --category pcb1
```

This repeats public benchmark **test images**, with mild simulated zoom and product
transitions. The video itself has a SYNTHETIC IMAGE DEMO banner outside the product ROI, so the watermark is not fed to the detector. Its source-image
list and synthetic status are preserved. It is not recorded motion, and no real-video
accuracy can be reported from it. Original benchmark licences continue to apply.

## Researched but not integrated as verified downloads

| Candidate | Temporal / annotation evidence | Access / integration decision |
|---|---|---|
| Iron-ore conveyor, DOI 10.17632/s25x2bnshz.1 | Publisher describes 19 original MP4s, 120 FPS; partially labeled extracted image splits, no complete video timestamp ground truth | CC BY 4.0 publisher page verified; live file APIs returned HTTP 403 in this environment. Not advertised as an automated working download. Public download page: https://data.mendeley.com/datasets/s25x2bnshz/1 |
| AeBAD-V | Recorded rotating blades; official repository describes video-level/sample-level evaluation | CC BY 4.0 stated by authors; full data linked through Drive/Tian Yi. Full archive access, exact annotation alignment and contents not verified here; not automatically downloaded. https://github.com/zhangzilongc/MMR |
| IPAD | Real/synthetic industrial temporal sequences, 16 devices, over six hours | Dataset linked from project page; page licence is not automatically the dataset licence. Redistribution terms and archive annotations not verified; no integration claim. https://ljf1113.github.io/IPAD_VAD/ |
| ConveyorAD | Conference abstract reports 140 annotated conveyor videos | No publicly downloadable archive verified. Not integrated. https://cs2026.sciencesconf.org/703307/document |

There is no labeled video benchmark integrated yet. This does not block video inference.
Adding a new scene-specific EfficientAD detector would require normal training frames;
CPU training may exceed two hours. No such training has been launched.
