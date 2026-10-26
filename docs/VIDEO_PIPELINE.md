# Streaming inspection and annotation policy

Input is an existing MP4/AVI/MOV, a prepared real conveyor clip, or the separately
identified synthetic image demonstration. The UI has no webcam/recording input.

Frames are decoded sequentially, bounded by duration and sampled-frame count. The
default UI processes five seconds and every fifteenth frame. At most 120 observations
are analyzed. Original input is never changed. Reports distinguish original duration
from processed duration and indicate truncation. Unsupported or undecodable files,
invalid FPS, failed codecs, missing checkpoints and inference errors fail explicitly.

Each analyzed frame gets a calibrated heatmap overlay and score. Unsampled frames
are marked NOT ANALYZED rather than inheriting a fabricated prediction. Per-frame
compressed NPZs retain raw/calibrated maps resized to 256x256, timestamps and ROI. A fixed ROI can be
specified as `x,y,width,height` in source coordinates; it is not object tracking.

Events first require consecutive suspicious sampled observations. Qualifying runs
can be merged across the configured gap. Minimum duration is optional. Events are
time intervals, **not counts of defective products**. No track IDs are invented.

Each run has a unique `artifacts/VID_...` directory containing:

- `annotated.mp4` (browser-compatible H.264/yuv420p via bundled FFmpeg).
- `frame_scores.csv`, `inspection_events.csv`, `inspection_report.json`.
- `maps/frame_*.npz` and a bounded gallery displayed in the UI.

If browser playback fails, use FFmpeg to convert to H.264:
`ffmpeg -i annotated.mp4 -c:v libx264 -pix_fmt yuv420p playable.mp4`.
The pinned imageio-ffmpeg dependency supplies a bundled encoder; no separate manual install is required.

## Optional existing ground truth

Without annotations, evaluation mode is qualitative and accuracy metrics are null.
Throughput and processing metadata remain available. Complete existing annotations
can be supplied to the CLI/UI as JSON:

```json
{
  "annotation_type": "temporal_intervals",
  "complete": true,
  "coverage_start_s": 0,
  "coverage_end_s": 20,
  "events": [{"start_time_s": 4.2, "end_time_s": 5.8}]
}
```

Outside marked events is treated as normal **only** when complete coverage is
explicitly declared. Partial annotations are rejected for quantitative evaluation.
Frame labels align to exact sampled timestamps, using half-open event intervals.
Frame AUROC requires both classes. Event precision/recall use one-to-one temporal
IoU matching (default 0.1); duplicate alerts are additional overlapping predictions,
not unrelated false positives. Mean delay uses the confirmation time of the minimum
consecutive run, rather than pretending the first suspicious frame was already a
confirmed alert, and is computed only for matched events.
Synthetic image sequences are rejected as real-video quantitative benchmarks.

No public ground-truth timestamps have been fabricated for the included conveyor clip.
