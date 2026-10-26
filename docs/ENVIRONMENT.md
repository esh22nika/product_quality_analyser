# Dependency environment

`requirements.txt` pins the upstream-supported target: Anomalib 2.7.0 with the
matched PyTorch 2.10.0 / torchvision 0.25.0 pair. Python 3.12 is the available local
interpreter; Python 3.11 is also a package-supported target, but was not used here.
Use a project virtual environment rather than upgrading another project's Python.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CUDA choose a matched torch/torchvision wheel pair from the official PyTorch
installer instead of the CPU index. No paid account is required for local CPU use.

The inherited global environment had torch 2.5.1 and torchvision 0.20.1 with
Anomalib 2.7.0. This is below Anomalib's official torch floor; earlier inference
checks passing in it do not make it a supported training environment.
`requirements-validated-local.txt` records that inherited environment for audit.
The project-local `.venv` isolates the supported torch pair while reusing already
installed Python dependencies during local validation. This is not a clean-room
installation test; fresh-install commands are documented separately above.
The full automated test suite passed in the supported project environment.

`imageio-ffmpeg==0.6.0` supplies the FFmpeg executable used to produce browser-playable
H.264 videos. No manual FFmpeg recording/setup is required. Native OpenCV decode or
writer failures return useful errors. Do not install an unpinned new torch release
independently of torchvision.
