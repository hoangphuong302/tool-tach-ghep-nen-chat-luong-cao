from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np

_NVENC_CHECKED: Optional[bool] = None


def _has_nvenc() -> bool:
    """Test once whether FFmpeg h264_nvenc actually initializes on this GPU."""
    global _NVENC_CHECKED
    if _NVENC_CHECKED is not None:
        return _NVENC_CHECKED
    try:
        ff = get_ffmpeg_exe()
        proc = subprocess.run(
            [
                ff,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                "256x256",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            input=b"\x00" * (256 * 256 * 3),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
        _NVENC_CHECKED = proc.returncode == 0
    except Exception:
        _NVENC_CHECKED = False
    return bool(_NVENC_CHECKED)


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def duration_sec(self) -> float:
        return float(self.frame_count) / max(float(self.fps or 30.0), 1.0)


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


def _write_extracted_frame(
    frame: np.ndarray,
    fp: Path,
    target_size: Optional[tuple[int, int]],
    ext: str,
    jpeg_quality: int,
) -> Path:
    if target_size is not None and (frame.shape[1], frame.shape[0]) != target_size:
        interp = cv2.INTER_AREA if (frame.shape[1] > target_size[0]) else cv2.INTER_CUBIC
        frame = cv2.resize(frame, target_size, interpolation=interp)
    if ext.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        cv2.imwrite(str(fp), frame)
    return fp


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_size: Optional[tuple[int, int]] = None,
    ext: str = ".jpg",
    jpeg_quality: int = 95,
) -> list[Path]:
    """Extract all frames sequentially from VideoCapture with bounded parallel JPEG writes."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for old in out_path.glob(f"*{ext}"):
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths: list[Path] = []
    pending: deque = deque()
    idx = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fp = out_path / f"{idx:05d}{ext}"
                frame_paths.append(fp)
                while len(pending) >= 12:
                    pending.popleft().result()
                pending.append(pool.submit(_write_extracted_frame, frame, fp, target_size, ext, jpeg_quality))
                idx += 1
            while pending:
                pending.popleft().result()
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


def load_alpha(path: str | Path) -> np.ndarray:
    """Load an 8-bit grayscale matte PNG as continuous float32 [0.0, 1.0]."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Alpha matte not found: {path}")
    return m.astype(np.float32) / 255.0


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


class VideoPipeEncoder:
    """
    Stream raw BGR frames straight into FFmpeg GPU h264_nvenc (with libx264 fallback)
    over a stdin pipe so zero intermediate PNG frames hit the disk during render.
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        audio_source_path: Optional[str | Path] = None,
        prefer_nvenc: bool = True,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps or 30.0)
        self._closed = False

        ffmpeg = get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",
            "-nostats",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
        ]
        if audio_source_path is not None and Path(audio_source_path).exists():
            cmd.extend(["-i", str(audio_source_path), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"])
        else:
            cmd.extend(["-an"])

        if (self.width % 2 != 0) or (self.height % 2 != 0):
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])

        if prefer_nvenc and min(self.width, self.height) >= 160 and _has_nvenc():
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p"])
        else:
            cmd.extend(["-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p"])

        cmd.extend(["-movflags", "+faststart", str(self.output_path)])
        self._stderr_file = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )

    def write(self, bgr_frame: np.ndarray) -> None:
        if self._closed or self._proc.stdin is None:
            raise RuntimeError("VideoPipeEncoder already closed.")
        if not bgr_frame.flags["C_CONTIGUOUS"] or bgr_frame.dtype != np.uint8:
            bgr_frame = np.ascontiguousarray(bgr_frame, dtype=np.uint8)
        self._proc.stdin.write(memoryview(bgr_frame))

    def close(self) -> Path:
        if self._closed:
            return self.output_path
        self._closed = True
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        rc = self._proc.wait()
        self._stderr_file.seek(0)
        stderr_bytes = self._stderr_file.read()
        self._stderr_file.close()
        if rc != 0:
            raise RuntimeError(f"FFmpeg pipe failed ({rc}):\n{stderr_bytes.decode('utf-8', 'replace')[-1500:]}")
        return self.output_path

    def __enter__(self) -> "VideoPipeEncoder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._closed = True
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            try:
                self._stderr_file.close()
            except Exception:
                pass
            return
        self.close()
