"""Browser-playable H.264 exports using the bundled, pinned FFmpeg executable."""
import os
from pathlib import Path
import subprocess


def browser_encode(path):
    import imageio_ffmpeg
    path = Path(path)
    converted = path.with_name(path.stem + '.h264.mp4')
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(path), '-an', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(converted)],
                       check=True, capture_output=True, timeout=180,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        os.replace(converted, path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        converted.unlink(missing_ok=True)
        raise RuntimeError('H.264 conversion failed; the intermediate mp4v video remains available on disk.') from exc
    return path
