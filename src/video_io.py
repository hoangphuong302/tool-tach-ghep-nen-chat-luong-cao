from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


def get_ffmpeg_exe() -> str:
    """Return system ffmpeg if available, else imageio-ffmpeg binary."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMetadata(fps=fps, width=width, height=height, frame_count=frame_count)
    finally:
        cap.release()


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_size: Optional[tuple[int, int]] = None,
    ext: str = ".jpg",
    jpeg_quality: int = 95,
) -> list[Path]:
    """Extract all frames to disk sequentially without holding video in RAM."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # Clean old frames if re-extracting
    for old in out_path.glob(f"*{ext}"):
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths: list[Path] = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if target_size is not None and (frame.shape[1], frame.shape[0]) != target_size:
                interp = cv2.INTER_AREA if (frame.shape[1] > target_size[0]) else cv2.INTER_CUBIC
                frame = cv2.resize(frame, target_size, interpolation=interp)
            fp = out_path / f"{idx:05d}{ext}"
            if ext.lower() in (".jpg", ".jpeg"):
                cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(fp), frame)
            frame_paths.append(fp)
            idx += 1
    finally:
        cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frame_paths


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(p), u8)


def load_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return (m > 127).astype(np.uint8)


def encode_video(
    frames_dir: str | Path,
    output_path: str | Path,
    fps: float,
    audio_source_path: Optional[str | Path] = None,
    pattern: str = "%05d.png",
) -> Path:
    """Encode rendered frames to MP4 (libx264, CRF 16, yuv420p, medium)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    input_pattern = str(Path(frames_dir) / pattern)

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        input_pattern,
    ]

    if audio_source_path is not None and Path(audio_source_path).exists():
        cmd.extend(["-i", str(audio_source_path), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"])
    else:
        cmd.extend(["-an"])

    cmd.extend([
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed ({proc.returncode}):\n{proc.stderr[-1500:]}")
    return out
